import os
import json
import logging
import ast
from typing import List, Dict, Tuple, Union, Any
import re
from typing import Any
from app.vector_db import VectorDB
from app.model_runner import ModelCandidate, ModelChoice

# Интеграция с локальным LLM через Hugging Face `transformers` (если задан HF_MODEL),
# иначе используется rules-based fallback. Это даёт лёгкую локальную альтернативу OpenAI/Ollama.

# Default behavior: use a rules-based fallback unless the user explicitly sets `HF_MODEL`.
# This avoids attempting to load large transformer models in test environments where the
# variable is not intentionally provided. If you want to use a local HF model set the
# `HF_MODEL` environment variable to the model name (e.g. "TinyLlama/TinyLlama_v1.1").
HF_MODEL_DEFAULT = "TinyLlama/TinyLlama_v1.1"

BASELINE_PROMPT = """
You are an assistant that helps choose the most suitable spatial-temporal model for a dataset.
Available candidates are provided as either library names or (model_name, library) pairs.

Examples of candidate formats:
- 'sktime'  # allow the selector to choose any suitable model from the sktime family
- ('AutoARIMA','sktime')  # explicitly offer AutoARIMA from sktime
- ('persistence','pysteps')
- ('GAT','torch_geometric')

User will provide:
- short description of dataset (spatial resolution, temporal frequency, number of sensors)
- task (nowcasting/reconstruction/forecasting)
- performance constraints (latency, memory)
 
Return a JSON with fields: model_choice, library, model_name (optional), model_args (optional dict), rationale, confidence (0-1).

Choose a single best option from the provided candidates and justify briefly. When a candidate specifies a concrete model (model_name, library), prefer that model if appropriate. If the candidate is only a library name, you may propose a concrete model_name from that library in the response.
"""


def build_prompt(dataset_desc: str, task: str, candidates: List[Union[str, Tuple[str, str]]]) -> str:
    # We add explicit markers so the model is instructed to place the JSON between
    # clear delimiters. This helps post-processing reliably extract JSON even when
    # the model emits surrounding explanation text.
    prompt = BASELINE_PROMPT + "\nUser-provided info:\n"
    prompt += f"Dataset: {dataset_desc}\n"
    prompt += f"Task: {task}\n"
    # Format candidates for prompt: represent pairs as "(model,library)" for clarity
    def _fmt(c):
        try:
            if isinstance(c, (list, tuple)):
                return f"({c[0]},{c[1]})"
        except Exception:
            pass
        return str(c)

    prompt += "Candidates: " + ", ".join([_fmt(c) for c in candidates]) + "\n"
    prompt += (
        "Please output only valid JSON as described, and place it between the markers "
        "<<JSON>> and <</JSON>> with no additional JSON-like text outside these markers."
    )
    return prompt


def call_transformers(prompt: str, model_name: str) -> Dict:
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError(f"transformers not available: {e}")

    # Попытка загрузить модель (может быть локальной) — задаётся через HF_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Try to import torch if available; it's optional for tests that mock transformers.
    try:
        import torch
        has_torch = True
    except Exception:
        torch = None  # type: ignore
        has_torch = False

    # Allow explicit override via environment variable HF_DEVICE:
    # -1 means CPU, 0 means first CUDA device, etc. Defaults to auto-detect.
    hf_dev = os.environ.get("HF_DEVICE", None)
    if hf_dev is not None:
        try:
            device = int(hf_dev)
        except Exception:
            device = -1
    else:
        if has_torch:
            try:
                device = 0 if getattr(torch, 'cuda', None) and torch.cuda.is_available() else -1
            except Exception:
                device = -1
        else:
            device = -1

    # Use logging for informational messages instead of printing to stdout
    logging.getLogger(__name__).info("Creating transformers pipeline (device=%s) for model %s", device, model_name)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    # Provide a small explicit example to encourage the model to place JSON between markers
    example_output = 'Example output:\n<<JSON>>{"library":"sktime","model_name":"AutoARIMA"}<</JSON>>\n'
    full_prompt = example_output + prompt
    out = gen(full_prompt, max_new_tokens=300, do_sample=False)
    # The generated output field depends on transformers version
    raw_text = out[0].get("generated_text") or out[0].get("text") or str(out[0])

    # Clean common wrappers/noise (device logs, warnings, code fences)
    text = _clean_transformers_text(raw_text)

    # Use shared extractor
    parsed = extract_json_from_text(text)
    if parsed is not None:
        return parsed

    # Try a tolerant eval-style fallback (single quotes, python dicts)
    try:
        maybe = ast.literal_eval(text.strip())
        if isinstance(maybe, dict):
            return maybe
    except Exception:
        pass
    # If still failing, attempt 1-2 retries with deterministic decoding (do_sample=False)
    retry_attempts = [
        {"do_sample": False},
        {"do_sample": False},
    ]
    stricter_prompt = (
        "Please output ONLY valid JSON between the markers <<JSON>> and <</JSON>>. "
        "Do not include any additional text outside the markers.\n" + example_output + prompt
    )
    attempt = 0
    for kw in retry_attempts:
        attempt += 1
        try:
            logging.getLogger(__name__).info('Retrying transformers generation attempt %d with %s', attempt, kw)
            out2 = gen(stricter_prompt, max_new_tokens=300, **kw)
            raw2 = out2[0].get('generated_text') or out2[0].get('text') or str(out2[0])
            logging.getLogger(__name__).debug('Retry raw output (attempt %d): %s', attempt, raw2[:400])
            text2 = _clean_transformers_text(raw2)
            parsed2 = extract_json_from_text(text2)
            if parsed2 is not None:
                return parsed2
            try:
                maybe2 = ast.literal_eval(text2.strip())
                if isinstance(maybe2, dict):
                    return maybe2
            except Exception:
                pass
            # save raw attempt for debugging
            try:
                dump_path = f'/tmp/llm_raw_output_attempt_{attempt}.txt'
                with open(dump_path, 'w', encoding='utf-8') as f:
                    f.write(raw2 if isinstance(raw2, str) else str(raw2))
                logging.getLogger(__name__).debug('Wrote retry raw transformers output to %s', dump_path)
            except Exception:
                dump_path = None
        except Exception as e:
            logging.getLogger(__name__).warning('Retry attempt %d failed: %s', attempt, e)

    # persist original raw output for offline inspection
    try:
        dump_path = '/tmp/llm_last_raw_output.txt'
        with open(dump_path, 'w', encoding='utf-8') as f:
            f.write(raw_text if isinstance(raw_text, str) else str(raw_text))
        logging.getLogger(__name__).debug('Wrote raw transformers output to %s', dump_path)
    except Exception:
        dump_path = None

    sample = text.strip()[:200].replace('\n', ' ')
    if dump_path:
        raise RuntimeError(
            f"Failed to parse JSON from local transformers output (sample={sample!r}); raw output saved to {dump_path}"
        )
    else:
        raise RuntimeError(f"Failed to parse JSON from local transformers output (sample={sample!r})")


def _clean_transformers_text(text: str) -> str:
    """Remove common wrappers and noise from transformer-generated text.

    - Strip device/warning lines (e.g., 'Device set to use cuda:0')
    - Remove markdown/code fences ```...``` and inline backticks
    - Collapse repeated whitespace
    """
    if not isinstance(text, str):
        return str(text)

    # Remove common device/warning lines
    lines = []
    for ln in text.splitlines():
        if re.search(r"device set to use", ln, flags=re.I):
            # drop diagnostic device messages
            continue
        if re.search(r"the following generation flags are not valid", ln, flags=re.I):
            continue
        lines.append(ln)
    text = "\n".join(lines)

    # Replace triple-backtick code fences and keep their inner content (commonly used for JSON)
    # e.g. ```json\n{...}\n``` -> {...}
    text = re.sub(r"```(?:json)?\n?(.*?)```", r"\1", text, flags=re.S)
    # Remove inline backticks
    text = text.replace('`', '')
    # Collapse multiple newlines and whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_json_from_text(text: str) -> Dict | None:
    """Extract JSON object from free-form model text.

    Returns parsed dict if successful, otherwise None.
    This pulls out JSON placed between <<JSON>><</JSON>> markers first, then
    searches for the first balanced JSON substring.
    """
    # Попробуем извлечь JSON между явными маркерами, если они есть
    marker_match = re.search(r"<<JSON>>(.*?)<</JSON>>", text, flags=re.S)
    if marker_match:
        candidate = marker_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            # fall through to balanced extraction
            pass

    # Поиск первой сбалансированной JSON-подстроки.
    # Try to find a balanced JSON object start..end; be tolerant to extra trailing text
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    # Try to repair JSON: convert single quotes to double quotes for keys/strings
                    repaired = candidate.replace("'", '"')
                    try:
                        return json.loads(repaired)
                    except Exception:
                        # give up on this candidate, but continue searching for another opening brace
                        pass
    # As a last resort, attempt to find any {...} substring via regex (non-greedy)
    for m in re.finditer(r"\{.*?\}", text, flags=re.S):
        s = m.group(0)
        try:
            return json.loads(s)
        except Exception:
            try:
                return json.loads(s.replace("'", '"'))
            except Exception:
                continue
    return None


def select_model(dataset_desc: str, task: str, candidates: List[Union[str, Tuple[str, str]]], hf_model_name: str = HF_MODEL_DEFAULT) -> Dict:
    # ensure we pass a list with the annotated union type to build_prompt
    prompt = build_prompt(dataset_desc, task, list(candidates))
    # Если задано имя HF-модели — попробуем локальную модель через transformers.
    # hf_model_name по умолчанию указывает на TinyLlama; если загрузка модели/библиотек
    # не удалась, мы безопасно откатимся к rules-based fallback.
    if hf_model_name:
        try:
            return call_transformers(prompt, hf_model_name)
        except Exception as e:
            print(f"Local LLM call failed for model '{hf_model_name}': {e}. Using fallback rules.")

    # Simple rules-based baseline (fallback)
    desc = (dataset_desc or "").lower()
    task_l = (task or "").lower()

    if "graph" in desc or "sensor network" in desc or "many sensors" in desc:
        return {
            "model_choice": "Graph-based GNN (GCN/GAT)",
            "library": "torch_geometric",
            "model_name": "GAT",
            "model_args": {"layers": 2},
            "rationale": "Dataset appears to have graph structure or many sensors, GNNs model spatial relations well.",
            "confidence": 0.7,
        }

    if "radar" in desc or "precip" in desc or "nowcast" in task_l:
        return {
            "model_choice": "Nowcasting (optical-flow / persistence)",
            "library": "pysteps",
            "model_name": "persistence",
            "model_args": {},
            "rationale": "Radar-based nowcasting algorithms like in PySTEPS are tailored for this task.",
            "confidence": 0.8,
        }

    if "shape" in desc or "dtw" in desc or "clustering" in task_l:
        return {
            "model_choice": "Distance-based / clustering",
            "library": "tslearn",
            "model_name": "KShape",
            "model_args": {"n_clusters": 8},
            "rationale": "Shape-based similarity and clustering methods are provided by tslearn.",
            "confidence": 0.6,
        }

    # default: sktime for time-series forecasting
    return {
        "model_choice": "Classical time-series or ML estimator",
        "library": "sktime",
        "model_name": "AutoARIMA",
        "model_args": {},
        "rationale": "sktime provides a broad set of forecasting and classical methods as a good baseline.",
        "confidence": 0.5,
    }


def select_model_rag(dataset_desc: str, task: str, db: Any = None, top_k: int = 4, hf_model_name: str = HF_MODEL_DEFAULT) -> Dict:
    """Retrieve candidate model descriptions from vector DB (RAG) then call select_model.

    - First perform a text search in `vector_db` for dataset/task description to get relevant
      implemented model descriptions.
    - Optionally restrict to provided `candidates` (if non-empty); candidates can be
      library names or (model_name, library) pairs.
    - Build an enriched candidates list where each candidate may include a 'description'
      field which will be included in the prompt for final selection.
    """
    # best-effort initialize DB
    # accept only VectorDB instances or None — do not rely on legacy module wrappers
    if db is None:
        db_client = VectorDB(backend='memory')
    elif isinstance(db, VectorDB):
        db_client = db
    else:
        # if a non-VectorDB object is provided, ignore it and create a transient DB
        db_client = VectorDB(backend='memory')
        try:
            db_client.initialize()
        except Exception:
            pass

    query_text = f"{dataset_desc} | task: {task}"
    docs = []
    try:
        docs = db_client.search_by_text(query_text, k=top_k)
    except Exception:
        docs = []

    # docs: list of (metadata, score)
    enriched: List[ModelCandidate] = []
    for md, score in docs:
        if not isinstance(md, dict):
            continue
        # md expected to have model_name, library, description
        name = md.get('model_name') or md.get('id') or ''
        lib = md.get('library') or ''
        desc = md.get('description', '')
        enriched.append(ModelCandidate({'model_name': name, 'library': lib, 'description': desc, 'id': md.get('id', ''), 'score': float(score)}))

    # If nothing found in vector DB, leave enriched empty — downstream fallback will handle it

    # Build a candidates arg for select_model: include descriptions to help LLM
    select_candidates: List[ModelCandidate] = []
    for e in enriched:
        name = e.get('model_name') or ''
        lib = e.get('library') or ''
        desc = e.get('description', '')
        select_candidates.append(ModelCandidate({'model_name': name, 'library': lib, 'description': desc, 'id': e.get('id',''), 'score': e.get('score', 0.0)}))

    # Build a prompt-friendly list where each candidate entry is represented as 'model_name (library): description'
    # We'll craft a short prompt addition and call the base select_model with the original candidates list
    # but we include descriptions in the global prompt by temporarily calling build_prompt with enriched formatting.
    prompt_extra = '\n\nAvailable implemented candidates (from DB):\n'
    for e in select_candidates:
        prompt_extra += f"- {e.get('model_name') or '<any>'} ({e.get('library')}): {e.get('description')}\n"

    # Fallback to existing select_model behaviour but prefer enriched candidates if possible
    # If enriched contains dicts with descriptions, we construct a simplified candidates list for select_model
    simple_candidates: List[Union[str, Tuple[str,str]]] = []
    for e in select_candidates:
        mname = str(e.get('model_name') or '')
        lib = str(e.get('library') or '')
        if mname:
            simple_candidates.append((mname, lib))
        else:
            simple_candidates.append(lib)

    # We will call call_transformers directly if HF_MODEL is set, otherwise use fallback select_model logic.
    try:
        # Compose a prompt merging baseline and extra info
        prompt = build_prompt(dataset_desc + '\n' + prompt_extra, task, simple_candidates)
        if hf_model_name:
            try:
                return call_transformers(prompt, hf_model_name)
            except Exception:
                # if local model call fails, fall back to rules-based selection
                pass
    except Exception:
        pass

    return select_model(dataset_desc, task, simple_candidates, hf_model_name=hf_model_name)
