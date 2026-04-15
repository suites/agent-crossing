from importlib import import_module
from typing import TypeVar, cast

LANGGRAPH_GRAPH_MODULE = import_module("langgraph.graph")
GRAPH_START = cast(object, getattr(LANGGRAPH_GRAPH_MODULE, "START"))
GRAPH_END = cast(object, getattr(LANGGRAPH_GRAPH_MODULE, "END"))


def _load_state_factory() -> object:
    return cast(object, getattr(LANGGRAPH_GRAPH_MODULE, "StateGraph"))


GRAPH_STATE_FACTORY = _load_state_factory()

TStateValue = TypeVar("TStateValue")


def require_state_value(value: TStateValue | None, *, key: str) -> TStateValue:
    if value is None:
        raise RuntimeError(f"Graph state is missing required value: {key}")
    return value
