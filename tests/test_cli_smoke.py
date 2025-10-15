import subprocess
import sys


def _run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def test_single_symbol_synthetic():
    rc = _run(
        [
            sys.executable,
            "-m",
            "trading_stack_py.cli",
            "--ticker",
            "RELIANCE.NS",
            "--start",
            "2015-01-01",
            "--source",
            "synthetic",
        ]
    )
    assert rc == 0


def test_portfolio_synthetic():
    rc = _run(
        [
            sys.executable,
            "-m",
            "trading_stack_py.portfolio_cli",
            "--tickers",
            "RELIANCE.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS,TCS.NS",
            "--start",
            "2018-01-01",
            "--source",
            "synthetic",
            "--lookback",
            "126",
            "--top_n",
            "3",
            "--cost_bps",
            "10",
        ]
    )
    assert rc == 0
