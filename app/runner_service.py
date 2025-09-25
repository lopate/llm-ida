from typing import Any, Dict, List, Optional
import numpy as np

from app.model_runner import run_model_from_choice, ModelChoice
from app import llm


def run_service(
    dataset_desc: str,
    input_data: Any,
    db: Any = None,
    task: str = 'forecast',
    horizon: int = 3,
    prefer_selector: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """High-level wrapper that selects a model via `app.llm.select_model` and runs it.

    New signature expects `input_data` as the second positional argument to match the
    demo notebooks: `run_service(dataset_desc, input_data, db=..., task=..., horizon=...)`.

    Returns a dict with keys: 'choice' (the selection dict returned by llm.select_model),
    'result' (runner output), and 'logs' (list[str]).
    """

    logs: List[str] = []

    choice: Any = None

    # First try the RAG-based selector from app.llm (vector DB retrieval + selection)
    try:
        sel = llm.select_model_rag(dataset_desc, task, db=db)
        if isinstance(sel, dict) and 'library' in sel:
            choice = sel
            logs.append('used app.llm.select_model_rag')
        else:
            choice = sel
            logs.append('used app.llm.select_model_rag (unverified dict)')
    except Exception as e:
        logs.append(f'app.llm.select_model_rag failed: {e}')

    # Fallback heuristic if selector didn't produce a choice
    if choice is None:
        try:
            arr = np.asarray(input_data)
            if arr.ndim == 1:
                choice = {'library': 'sktime', 'model_name': 'auto_arima', 'model_args': {}}
            elif arr.ndim == 2:
                # (T, N) -> sensors
                choice = {'library': 'tslearn', 'model_name': 'knn', 'model_args': {'n_neighbors': 3}}
            else:
                choice = {'library': 'pysteps', 'model_name': 'persistence', 'model_args': {}}
            logs.append('used heuristic choice based on ndarray ndim')
        except Exception as e:
            choice = {'library': 'sktime', 'model_name': 'auto_arima', 'model_args': {}}
            logs.append(f'heuristic failed, defaulting to sktime: {e}')

    # Ensure choice is a plain dict and coerce to ModelChoice (typed) for runner
    if not isinstance(choice, dict):
        try:
            choice = dict(choice)
        except Exception:
            choice = {'library': 'sktime', 'model_name': 'auto_arima', 'model_args': {}}
            logs.append('coerced choice to default dict')

    # Build a ModelChoice safely
    safe_choice: ModelChoice = ModelChoice({
        'library': str(choice.get('library') or choice.get('model_choice') or 'sktime'),
    })
    if 'model_name' in choice and choice.get('model_name'):
        safe_choice['model_name'] = str(choice.get('model_name'))
    if 'model_args' in choice:
        ma = choice.get('model_args') or {}
        if isinstance(ma, dict):
            safe_choice['model_args'] = ma
    choice = safe_choice

    # Run the selected model
    try:
        # run_model_from_choice expects ModelChoice|str; choice is runtime-dict from selector
        result = run_model_from_choice(choice, input_data, horizon=horizon, **kwargs)
    except Exception as exc:
        logs.append(f'runner_failed:{exc}')
        result = {
            'library': choice.get('library') if isinstance(choice, dict) else None,
            'y_pred': None,
            'csv': '',
            'meta': {'horizon': horizon},
            'error': str(exc),
        }

    return {'choice': choice, 'result': result, 'logs': logs}
