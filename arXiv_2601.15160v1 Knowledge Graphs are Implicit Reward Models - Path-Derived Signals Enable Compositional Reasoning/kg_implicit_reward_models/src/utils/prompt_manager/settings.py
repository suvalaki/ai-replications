from __future__ import annotations

from typing import Any, Optional, get_args, get_origin, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LangfusePromptManagerSettings(BaseSettings):
    prompt_name: str = Field(..., description="Langfuse prompt name")

    label: Optional[str] = "production"
    version: Optional[int] = None
    cache_ttl_seconds: int = 300
    fail_silent: bool = True
    create_missing: bool = True

    model_config = SettingsConfigDict(frozen=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        config = getattr(cls, "model_config", None)
        if not config or not config.get("frozen", False):
            raise TypeError(f"{cls.__name__} must have model_config with frozen=True")

    def __class_getitem__(cls, item: Any):
        # Accept both "foo" and Literal["foo"]
        if isinstance(item, str):
            prompt_name = item
        else:
            origin = get_origin(item)
            if origin is Literal:
                vals = get_args(item)
                if len(vals) != 1:
                    raise TypeError("Literal must contain exactly one value.")
                prompt_name = vals[0]
            else:
                raise TypeError(
                    "LangfusePromptManagerSettings[...] requires a string "
                    "or Literal['prompt-name']."
                )

        if not isinstance(prompt_name, str) or not prompt_name.strip():
            raise TypeError("Prompt name must be a non-empty string.")

        env_prefix = prompt_name.upper().replace("-", "_") + "_"

        # If you want the shorthand class name:
        safe = prompt_name.replace("-", "_")
        class_name = f"{safe}_LangfusePromptManagerSettings"
        # Or keep the old name:
        # class_name = f"{cls.__name__}[{prompt_name}]"

        namespace = {
            "__annotations__": {"prompt_name": str},
            "prompt_name": Field(default=prompt_name),
            "model_config": SettingsConfigDict(frozen=True, env_prefix=env_prefix),
        }
        return type(class_name, (cls,), namespace)
