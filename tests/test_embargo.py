from trading_stack_py.cv.walkforward import WalkForwardCV


def test_walkforward_embargo_shortens_train():
    n = 150
    cv_no = WalkForwardCV(
        train_size=60, test_size=15, step_size=15, expanding=True, embargo=0
    )
    cv_em = WalkForwardCV(
        train_size=60, test_size=15, step_size=15, expanding=True, embargo=5
    )
    tr0_no, te0_no = next(iter(cv_no.split(n)))
    tr0_em, te0_em = next(iter(cv_em.split(n)))
    assert len(tr0_em) < len(tr0_no)
    assert te0_em[0] == te0_no[0]  # embargo must not shift test
