from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    cast,
    overload,
)
from typing_extensions import Self, TypeGuard

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder, MessageLikeRepresentation
from pydantic.fields import ModelPrivateAttr

from langfuse import get_client
from langchain_core.prompts.string import PromptTemplateFormat
from langfuse.api.resources.commons.errors import NotFoundError

from .settings import LangfusePromptManagerSettings

logger = logging.getLogger(__name__)

LCMessageRep = MessageLikeRepresentation
RoleContent = Tuple[str, Any]


def _langchain_role_to_langfuse(role: str) -> str:
    r = role.lower().strip()
    if r in {"human", "user"}:
        return "user"
    if r in {"ai", "assistant"}:
        return "assistant"
    if r == "system":
        return "system"
    return r


def _is_role_content(x: object) -> TypeGuard[RoleContent]:
    return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str)


def _messages_to_langfuse_chat(
    messages: Sequence[LCMessageRep],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, MessagesPlaceholder):
            out.append({"role": "placeholder", "content": m.variable_name})
            continue

        if not _is_role_content(m):
            raise TypeError(f"Unsupported message rep: {type(m).__name__}: {m!r}")

        role, content = m
        out.append(
            {
                "role": _langchain_role_to_langfuse(role),
                "content": str(content).replace("{", "{{").replace("}", "}}"),
            }
        )
    return out


def _role_to_langchain(role: str) -> str:
    r = role.lower().strip()
    if r in {"user", "human"}:
        return "human"
    if r in {"assistant", "ai"}:
        return "ai"
    if r == "system":
        return "system"
    return r


def _convert_vars(s: str) -> str:
    return s.replace("{{", "{").replace("}}", "}")


def _extract_langfuse_chat_messages(langfuse_prompt: Any) -> List[Dict[str, Any]]:
    for attr in ("prompt", "messages", "chat", "content"):
        if hasattr(langfuse_prompt, attr):
            val = getattr(langfuse_prompt, attr)
            if isinstance(val, list) and (not val or isinstance(val[0], dict)):
                return val
    raise TypeError(
        "Unable to locate chat messages on langfuse_prompt (expected list[dict])."
    )


def _langfuse_messages_to_langchain(
    messages: Sequence[Dict[str, Any]],
) -> List[LCMessageRep]:
    out: List[LCMessageRep] = []
    for m in messages:
        role = str(m.get("role", "")).lower().strip()
        content = m.get("content", "")
        if role == "placeholder":
            out.append(MessagesPlaceholder(variable_name=str(content)))
            continue
        out.append((_role_to_langchain(role), _convert_vars(str(content))))
    return out


def _create_langfuse_chat_prompt(
    client: Any, *, name: str, label: str, messages: List[Dict[str, Any]]
) -> Any:
    """
    Create a chat prompt in Langfuse from messages payload.

    Adjust this adapter once to your SDK's actual signature if needed.
    """
    if not hasattr(client, "create_prompt"):
        raise AttributeError(
            "Langfuse client has no create_prompt(). Check your langfuse SDK version."
        )

    try:
        return client.create_prompt(
            name=name, prompt=messages, label=label, type="chat"
        )
    except TypeError:
        # Alternate common signature style
        return client.create_prompt(
            name=name, prompt=messages, labels=[label], type="chat"
        )


def _is_settings_cls(x: object) -> TypeGuard[type[LangfusePromptManagerSettings]]:
    return isinstance(x, type) and issubclass(x, LangfusePromptManagerSettings)


class LangfuseChatPromptTemplate(ChatPromptTemplate):
    """
    ChatPromptTemplate subtype with Langfuse-managed classmethod constructor.

    Usage:
      template = LangfuseChatPromptTemplate[KGQAMcqPromptSettings].from_messages(
          [...])
    """

    _settings_cls: Optional[Type[LangfusePromptManagerSettings]] = None

    def __class_getitem__(
        cls: type[Self],
        settings_cls: type[Any] | tuple[type[Any], ...],
    ) -> type[Any]:
        # normalize tuple form if you want to support PEP 560 edge-cases
        if isinstance(settings_cls, tuple):
            if len(settings_cls) != 1:
                raise TypeError("Expected a single settings class type parameter.")
            settings_cls = settings_cls[0]

        if not _is_settings_cls(settings_cls):
            raise TypeError(
                "Type parameter must subclass LangfusePromptManagerSettings."
            )

        name = f"{cls.__name__}[{settings_cls.__name__}]"
        return type(
            name,
            (cls,),
            {"__module__": cls.__module__, "_settings_cls": settings_cls},
        )

    @overload
    @classmethod
    def from_messages(
        cls,
        messages: Sequence[LCMessageRep],
        template_format: PromptTemplateFormat = "f-string",
    ) -> "LangfuseChatPromptTemplate": ...

    @classmethod
    @overload
    def from_messages(
        cls,
        messages: Sequence[LCMessageRep],
        template_format: PromptTemplateFormat = "f-string",
        langfuse_prompt_settings: Optional[LangfusePromptManagerSettings] = None,
        langfuse_client: Optional[Any] = None,
    ) -> "LangfuseChatPromptTemplate": ...

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[LCMessageRep],
        template_format: PromptTemplateFormat = "f-string",
        langfuse_prompt_settings: Optional[LangfusePromptManagerSettings] = None,
        langfuse_client: Optional[Any] = None,
    ) -> "LangfuseChatPromptTemplate":

        if langfuse_prompt_settings is None:
            if cls._settings_cls is None:
                raise TypeError(
                    "No settings bound. Use LangfuseChatPromptTemplate[MySettings].from_messages(...) "
                    "or pass langfuse_prompt_settings=MySettings()."
                )
            settings = cast(ModelPrivateAttr, cls._settings_cls).get_default()()
        else:
            settings = langfuse_prompt_settings

        client = langfuse_client or get_client()
        prompt_name = settings.prompt_name

        fallback = cast(
            LangfuseChatPromptTemplate,
            super().from_messages(list(messages)),
        )

        try:
            langfuse_prompt = client.get_prompt(
                prompt_name,
                label=settings.label,
                version=settings.version,
                cache_ttl_seconds=settings.cache_ttl_seconds,
            )

        except NotFoundError as e:
            # 🔴 ONLY AUTOCREATE WHEN SETTINGS ENABLE IT
            if not settings.create_missing:
                if not settings.fail_silent:
                    raise
                logger.warning(
                    "Langfuse prompt '%s' not found (%s). Using fallback.",
                    prompt_name,
                    str(e),
                )
                fallback.metadata = {
                    **(fallback.metadata or {}),
                    "langfuse_prompt_fallback": prompt_name,
                }
                return fallback

            # ✅ Auto-create path
            try:
                payload = _messages_to_langfuse_chat(messages)

                _create_langfuse_chat_prompt(
                    client,
                    name=prompt_name,
                    label=settings.label or "production",
                    messages=payload,
                )

                langfuse_prompt = client.get_prompt(
                    prompt_name,
                    label=settings.label,
                    version=settings.version,
                    cache_ttl_seconds=settings.cache_ttl_seconds,
                )

                logger.info("Langfuse prompt '%s' created automatically.", prompt_name)

            except Exception as create_exc:
                if not settings.fail_silent:
                    raise
                logger.warning(
                    "Langfuse prompt '%s' auto-create failed (%s). Using fallback.",
                    prompt_name,
                    str(create_exc),
                )
                fallback.metadata = {
                    **(fallback.metadata or {}),
                    "langfuse_prompt_fallback": prompt_name,
                }
                return fallback

        except Exception as e:
            if not settings.fail_silent:
                raise
            logger.warning(
                "Langfuse chat prompt '%s' unavailable (%s). Using fallback.",
                prompt_name,
                str(e),
            )
            fallback.metadata = {
                **(fallback.metadata or {}),
                "langfuse_prompt_fallback": prompt_name,
            }
            return fallback

        # ✅ Success path
        lf_messages = _extract_langfuse_chat_messages(langfuse_prompt)
        lc_messages = _langfuse_messages_to_langchain(lf_messages)

        managed = cast(
            LangfuseChatPromptTemplate,
            super().from_messages(lc_messages),
        )

        managed.metadata = {
            **(fallback.metadata or {}),
            "langfuse_prompt": langfuse_prompt,
        }

        return managed


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    z = LangfusePromptManagerSettings["kgqa-mcq-chat"]
    template = LangfuseChatPromptTemplate[z].from_messages(
        [
            ("system", "hi there"),
            ("ai", "this is an ai messsage"),
            ("human", "this is a human message"),
        ]
    )
