import pytest

from dgse_llm.models import build_model
from dgse_llm.schemas import ModelConfig


def pytest_addoption(parser):
    parser.addoption("--run-live", action="store_true", default=False, help="run tests that call real LLM APIs")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip = pytest.mark.skip(reason="needs --run-live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def ks_model():
    """Steady state of the Krusell-Smith model is expensive, so it is solved once per session."""
    return build_model(ModelConfig(name="krusell_smith"))


@pytest.fixture(scope="session")
def models(ks_model):
    """name -> model instance, shared across tests (steady states are cached on the instance)."""
    from dgse_llm.models import MODEL_REGISTRY

    out = {name: build_model(ModelConfig(name=name)) for name in MODEL_REGISTRY.names() if name != "krusell_smith"}
    out["krusell_smith"] = ks_model
    return out
