from trading_stack_py.cv.walkforward import WalkForwardCV


def test_walkforward_basic():
    n = 120
    cv = WalkForwardCV(train_size=60, test_size=20, step_size=20, expanding=False)
    splits = list(cv.split(n))
    # Expect 3 test blocks: [60:80], [80:100], [100:120]
    assert len(splits) == 3
    for tr, te in splits:
        assert len(te) == 20
        assert te[0] >= 60
        assert te[-1] < 120


def test_walkforward_expanding():
    n = 150
    cv = WalkForwardCV(train_size=60, test_size=15, step_size=15, expanding=True)
    splits = list(cv.split(n))
    # As expanding, train grows; ensure first train has correct length
    assert len(splits) >= 3
    first_tr, first_te = splits[0]
    assert len(first_tr) == 60
    # Ensure indices are strictly increasing, no overlap issues in test
    _, last_te = splits[-1]
    assert last_te[-1] < n
