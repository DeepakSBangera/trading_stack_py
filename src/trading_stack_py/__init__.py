# trading_stack_py package

try:
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover
    version = None
    PackageNotFoundError = Exception  # type: ignore

__all__ = ["__version__"]
try:
    __version__ = version("trading-stack-py")  # set your project name if different
except Exception:
    __version__ = "0.0.0+local"
