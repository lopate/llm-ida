import os
import json
from typing import List, Dict
import re

# Интеграция с локальным LLM через Hugging Face `transformers` (если задан HF_MODEL),
# иначе используется rules-based fallback. Это даёт лёгкую локальную альтернативу OpenAI/Ollama.

# Default behavior: use a rules-based fallback unless the user explicitly sets `HF_MODEL`.
# This avoids attempting to load large transformer models in test environments where the
# variable is not intentionally provided. If you want to use a local HF model set the
# `HF_MODEL` environment variable to the model name (e.g. "TinyLlama/TinyLlama_v1.1").
HF_MODEL = os.environ.get("HF_MODEL", "fallback")

BASELINE_PROMPT = """
You are an assistant that helps choose the most suitable spatial-temporal model for a dataset.
Available candidate libraries and models:
- pysteps: classical radar-based nowcasting approaches
- sktime: a variety of time-series models (ARIMA, ExponentialSmoothing, ML estimators)
- tslearn: distance-based and clustering methods for time-series
- torch_geometric: graph-based deep learning models (GCN, GAT, ST-Transformer-like)

User will provide:
- short description of dataset (spatial resolution, temporal frequency, number of sensors)
- task (nowcasting/reconstruction/forecasting)
- performance constraints (latency, memory)
 
 Return a JSON with fields: model_choice, library, model_name (optional), model_args (optional dict), rationale, confidence (0-1).

 Choose a single best option from candidates and justify briefly. When possible include a concrete `model_name` (for example: "AutoARIMA", "AutoETS", "Naive", "RandomForestReduction") and small `model_args` specifying notable hyperparameters (e.g. {"n_estimators":50}).
"""


def build_prompt(dataset_desc: str, task: str, candidates: List[str]) -> str:
    # We add explicit markers so the model is instructed to place the JSON between
    # clear delimiters. This helps post-processing reliably extract JSON even when
    # the model emits surrounding explanation text.
    prompt = BASELINE_PROMPT + "\nUser-provided info:\n"
    prompt += f"Dataset: {dataset_desc}\n"
    prompt += f"Task: {task}\n"
    prompt += "Candidates: " + ", ".join(candidates) + "\n"
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


def select_model(dataset_desc: str, task: str, candidates: List[str]) -> Dict:
    prompt = build_prompt(dataset_desc, task, candidates)

    # Если задан HF_MODEL — попробуем локальную модель через transformers.
    if HF_MODEL and HF_MODEL != "fallback":
        try:
            return call_transformers(prompt, HF_MODEL)
        except Exception as e:
            print(f"Local LLM call failed: {e}. Using fallback rules.")

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
