import app.model_runner as mr


def test_gcn_dummy():
    # simple smoke: ensure function callable
    out = mr.run_model_from_choice('torch_geometric', "t,value\n0,1\n1,2\n", horizon=1)
    assert out is not None
