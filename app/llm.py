import os
import json
from typing import List, Dict, Tuple, Union, Any
import re
from typing import Any
from app import vector_db
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
        import torch
    except Exception as e:
        raise RuntimeError(f"transformers/torch not available: {e}")

    # Попытка загрузить модель (может быть локальной) — задаётся через HF_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    out = gen(prompt, max_new_tokens=300, do_sample=False)
    # Выход может быть в поле 'generated_text' или 'text' в зависимости от версии
    text = out[0].get("generated_text") or out[0].get("text") or str(out[0])

    # Попробуем извлечь JSON между явными маркерами, если они есть
    marker_match = re.search(r"<<JSON>>(.*?)<</JSON>>", text, flags=re.S)
    if marker_match:
        candidate = marker_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            # fall through to balanced extraction
            pass

    # Если маркеры не найдены или парсинг не удался, используем предыдущую попытку —
    # поиск первой сбалансированной JSON-подстроки.
    def _extract_balanced_json(s: str) -> str:
        start = s.find('{')
        if start == -1:
            return ''
        depth = 0
        for i in range(start, len(s)):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return ''

    candidate = _extract_balanced_json(text)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Если всё ещё не получилось — вернём ошибку для fallback
    raise RuntimeError("Failed to parse JSON from local transformers output")


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
    # use provided db or default module vector_db
    db_client = db or vector_db
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
