from .processor import GenShell

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ['GenShell', '__version__']
