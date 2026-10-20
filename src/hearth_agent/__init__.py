"""Hearth: a private, local AI assistant for your terminal."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hearth-agent")
except PackageNotFoundError:
    __version__ = "0.0.0"
