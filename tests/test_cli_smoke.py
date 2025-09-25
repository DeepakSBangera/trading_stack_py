from trading_stack_py.cv.walkforward import WalkForwardCV


def test_wf_cli_shapes():
    n = 300
    cv = WalkForwardCV(train_size=120, test_size=30, step_size=30, expanding=False, embargo=3)
    splits = list(cv.split(n))
    assert len(splits) == 5
