def test_select_and_predict_fallback():
    from app.main import select_and_predict

    csv = "time,value\n0,1.0\n1,2.0\n2,3.0\n"
    class R:
        dataset_csv = csv
        dataset_description = "Sensor network with many nodes and graph topology"
        task = "forecasting"

    res = select_and_predict(R())
    assert isinstance(res, dict)
    assert "code_py" in res
    assert "predictions_csv" in res
    assert res.get("selected") is not None
