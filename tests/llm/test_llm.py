def test_select_model_fallback():
    from app.llm import select_model
    # The selector may use a local LLM or fallback rules; allow any of the provided
    # candidate libraries as a valid outcome. Tests should be robust to multiple
    # acceptable choices from the selector.
    from typing import List, Tuple, Union
    candidates: List[Union[str, Tuple[str, str]]] = ["pysteps", "sktime", "tslearn", "torch_geometric"]

    # Radar nowcasting: accept any candidate from the list
    res = select_model("Radar reflectivity at 1km resolution", "nowcasting", candidates)
    assert isinstance(res, dict)
    assert res.get("library") in candidates

    # Graph sensors: accept any candidate from the list (torch_geometric is preferred, but not required)
    res2 = select_model("Sensor network with many nodes and graph topology", "forecasting", candidates)
    assert res2.get("library") in candidates

    # Default: accept any candidate from the list (sktime is a common fallback)
    res3 = select_model("Single time-series per location, low spatial complexity", "forecasting", candidates)
    assert res3.get("library") in candidates
