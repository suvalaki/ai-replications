from typing import TypedDict, Optional, Any, cast, Callable
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import MessageLikeRepresentation
from src.utils.prompt_manager.chat_prompt_template import (
    LangfuseChatPromptTemplate,
    LangfusePromptManagerSettings,
)
from src.kgqa.schemas import QASpec
from src.kgqa.generation.base import MCQGenerateSettings


class PromptFromSpecGenerationInput(TypedDict):
    n_options: int
    option_labels: str
    chain: str
    rels: str


QAPromptBuilder = Runnable[QASpec, MessageLikeRepresentation]

SpecPromptSettings = LangfusePromptManagerSettings["spec-prompt"]


def default_prompt_from_spec_template(
    langfuse_prompt_settings: SpecPromptSettings | None = None,
    langfuse_client: Optional[Any] = None,
) -> LangfuseChatPromptTemplate:
    """The goal of this is to create a `prompt` for a MCQItem.

    We use LLMs to generate a prompt given the pre-existing reasoning chain.
    """
    return LangfuseChatPromptTemplate[SpecPromptSettings].from_messages(
        [
            (
                "system",
                "You are generating a multiple-choice question based on a knowledge graph "
                "reasoning chain. Based on the concept chain and the relations used: "
                "Write a clear question whose correct answer is the final concept in the chain.\n"
                "Return {n_options} labelled {option_labels} with exactly one correct.",
            ),
            (
                "ai",
                "Concept chain (do not reveal explicitly in the question): {chain}\n"
                "Relations used: {rels}\n\n",
            ),
        ],
        langfuse_prompt_settings=langfuse_prompt_settings,
        langfuse_client=langfuse_client,
    )


def default_extract_formatted_variables_from_spec(
    spec: QASpec,
    settings: MCQGenerateSettings | None = None,
) -> PromptFromSpecGenerationInput:
    _settings = settings or MCQGenerateSettings()
    return {
        "n_options": len(_settings.labels),
        "option_labels": str(_settings.labels),
        "chain": " -> ".join(
            [spec.start_text] + spec.intermediate_texts + [spec.answer_text]
        ),
        "rels": ", ".join(spec.relations),
    }


def default_prompt_from_spec_chain(
    llm: BaseChatModel,
    settings: MCQGenerateSettings | None = None,
    transform: (
        Callable[[QASpec, MCQGenerateSettings | None], PromptFromSpecGenerationInput]
        | None
    ) = None,
    langfuse_prompt_settings: SpecPromptSettings | None = None,
    langfuse_client: Optional[Any] = None,
    **kwargs: Any,
) -> QAPromptBuilder:

    return (
        RunnableLambda(
            (
                (lambda x: transform(x, settings))
                if transform
                else cast(
                    Callable[[QASpec], PromptFromSpecGenerationInput],
                    lambda x: default_extract_formatted_variables_from_spec(
                        x, settings
                    ),
                )
            ),
            name="InputTransformation",
        )
        | default_prompt_from_spec_template(
            langfuse_prompt_settings=langfuse_prompt_settings,
            langfuse_client=langfuse_client,
        )
        | llm
    ).with_config({"run_name": "DefaultQAPromptBuilder"})


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    load_dotenv()

    # Get the configured client instance
    langfuse = get_client()

    # Initialize the Langfuse handler
    langfuse_handler = CallbackHandler()

    llm = llm = ChatOpenAI(
        api_key=lambda: str(os.getenv("OPENROUTER_API_KEY")),
        base_url="https://openrouter.ai/api/v1",
        model=str(os.getenv("OPENROUTER_MODEL")),
        default_headers={},
    )

    spec = QASpec(
        **{
            "start_node_id": "DOID:9352",
            "answer_node_id": "DB01234",
            "start_text": "Type 2 Diabetes Mellitus",
            "answer_text": "Metformin",
            "relations": ["associates_with", "treated_by"],
            "intermediate_node_ids": ["GENE:4609"],
            "intermediate_texts": ["AMPK"],
            "ground_truth_path": [
                ["DOID:9352", "associates_with", "GENE:4609"],
                ["GENE:4609", "treated_by", "DB01234"],
            ],
            "hop_length": 2,
        }
    )

    chain = default_prompt_from_spec_chain(llm)

    chain.invoke(spec, config={"callbacks": [langfuse_handler]})
