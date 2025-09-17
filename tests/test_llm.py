def test_select_model_fallback():
    from app.llm import select_model

    # Radar nowcasting -> expect pysteps
    res = select_model("Radar reflectivity at 1km resolution", "nowcasting", ["pysteps","sktime","tslearn","torch_geometric"])
    assert isinstance(res, dict)
    assert res.get("library") == "pysteps"

    # Graph sensors -> expect torch_geometric
    res2 = select_model("Sensor network with many nodes and graph topology", "forecasting", ["pysteps","sktime","tslearn","torch_geometric"])
    assert res2.get("library") == "torch_geometric"

    # Default -> sktime
    res3 = select_model("Single time-series per location, low spatial complexity", "forecasting", ["pysteps","sktime","tslearn","torch_geometric"])
    assert res3.get("library") == "sktime"
