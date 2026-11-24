import importlib
import re
from pathlib import Path

from prefect import Flow

ROOT = Path(__file__).resolve().parents[2]


def flows_in_repo():
    found = {}
    for path in sorted((ROOT / "src/flows").glob("*.py")):
        module = importlib.import_module(f"src.flows.{path.stem}")
        for name, value in vars(module).items():
            if isinstance(value, Flow) and value.__module__ == module.__name__:
                found[name] = path.name
    return found


def test_every_flow_imports_and_has_a_test():
    flows = flows_in_repo()
    assert flows, "no flows found"
    tests = "\n".join(p.read_text() for p in Path(__file__).parent.glob("test_*.py") if p.name != Path(__file__).name)
    untested = [f"{name} ({file})" for name, file in flows.items() if not re.search(rf"\b{name}\(", tests)]
    assert not untested, f"flows with no test in tests/flows/: {untested}"
