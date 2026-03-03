from functools import lru_cache
from pathlib import Path
from string import Template

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=64)
def _load_template_text(template_name: str) -> str:
    path = TEMPLATE_DIR / template_name
    return path.read_text(encoding="utf-8")


def render_template(template_name: str, **variables: str) -> str:
    template_text = _load_template_text(template_name)
    template = Template(template_text)
    try:
        return template.substitute(**variables)
    except KeyError as exc:
        missing = str(exc)
        raise ValueError(
            f"Missing template variable {missing} for {template_name}"
        ) from exc
