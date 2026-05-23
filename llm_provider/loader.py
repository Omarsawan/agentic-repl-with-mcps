import importlib.util
import os

from .base import LLMProvider
from .keyword_match import KeywordMatchProvider
from .openai_compatible import OpenAICompatibleProvider


def build_provider(config: dict) -> LLMProvider:
    """Construct an LLMProvider from the 'llm' section of mcp_servers.json.

    Supported provider types:
    - "openai_compatible" (default): uses OpenAICompatibleProvider.
    - "keyword_match": uses KeywordMatchProvider (no LLM required).
    - "custom": dynamically loads a LLMProvider subclass via _load_custom_provider().
    """
    provider_type = config.get("provider", "openai_compatible")

    if provider_type == "openai_compatible":
        base_url = config.get("base_url", "http://localhost:11434/v1")
        model = config.get("model", "qwen2.5:7b")
        api_key_env = config.get("api_key_env", "LLM_API_KEY")
        api_key = os.environ.get(api_key_env, "ollama")
        return OpenAICompatibleProvider(base_url=base_url, model=model, api_key=api_key)

    if provider_type == "keyword_match":
        return KeywordMatchProvider()

    if provider_type == "custom":
        return _load_custom_provider(config)

    raise ValueError(f"Unknown provider type: {provider_type!r}")


def _load_custom_provider(config: dict) -> LLMProvider:
    """Dynamically import and instantiate a user-supplied LLMProvider subclass.

    Reads "module" (path to a .py file) and "class" (class name) from the config.
    The class must be a subclass of LLMProvider. All remaining config keys are passed
    as keyword arguments to the constructor; "api_key_env" is resolved to its env value first.
    """
    module_path = config.get("module")
    class_name = config.get("class")
    if not module_path or not class_name:
        raise ValueError("Custom provider requires 'module' and 'class' in the llm config.")

    spec = importlib.util.spec_from_file_location("_custom_provider", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Class {class_name!r} not found in {module_path!r}")
    if not (isinstance(cls, type) and issubclass(cls, LLMProvider)):
        raise TypeError(f"{class_name!r} must be a subclass of LLMProvider")

    kwargs = {k: v for k, v in config.items() if k not in {"provider", "module", "class"}}
    if "api_key_env" in kwargs:
        kwargs["api_key"] = os.environ.get(kwargs.pop("api_key_env"), "")

    return cls(**kwargs)
