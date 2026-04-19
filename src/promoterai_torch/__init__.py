from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("promoterai-torch")
except PackageNotFoundError:
    __version__ = "unknown"
