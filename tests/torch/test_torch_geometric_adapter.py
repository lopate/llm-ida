import app.model_runner as mr


def test_pysteps_basic_multi_node():
    # build simple CSV with t,node,value for T=5, N=2
    lines = [["t", "node", "value"]]
    for t in range(5):
        for n in range(2):
            lines.append([str(t), str(n), f"{t*10 + n}.0"])
    csv_text = "\n".join([",".join(r) for r in lines])

    out = mr.run_pysteps(csv_text, horizon=3)
    assert out is not None
