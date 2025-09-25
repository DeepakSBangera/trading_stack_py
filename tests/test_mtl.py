import math

from trading_stack_py.metrics.mtl import minimum_track_record_length


def test_mtl_monotonicity_in_pstar():
    sr = 0.5
    n90 = minimum_track_record_length(sr_hat=sr, p_star=0.90)
    n95 = minimum_track_record_length(sr_hat=sr, p_star=0.95)
    assert n95 >= n90 > 1


def test_mtl_infinite_when_sr_below_threshold():
    n = minimum_track_record_length(sr_hat=0.0, sr_threshold=0.1)
    assert math.isinf(n)
