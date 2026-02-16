from typing import Dict, TypedDict, Any
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic_settings import BaseSettings


class MCQGenerateSettings(BaseSettings):
    labels: list[str] = ["A", "B", "C", "D"]


class MCQGeneratorInputs(TypedDict):
    prompt: str
    correct_answer: str
    distractors: Dict[str, str]
    labels: list[str]


class MCQDraft(TypedDict):
    question: str
    options: Dict[str, str]  # keys: "A","B","C","D"
    correct_option: str  # "A".."D"


MCQGenerator = Runnable[MCQGeneratorInputs, MCQDraft]


class SimpleTemplateMCQGenerator(MCQGenerator):
    """Dummy runnable that always deterministically sets the
    correct anser to be A. For testing purposes."""

    def invoke(
        self,
        input: MCQGeneratorInputs,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> MCQDraft:
        # Very simple, not "good", but keeps pipeline runnable.
        options = {input.labels[0]: input.correct_answer}
        labels = input.labels
        for lab, txt in zip(labels, input.distractors.values()):
            options[lab] = txt

        return MCQDraft(
            question=input.prompt.strip(),
            options=options,
            correct_option="A",
        )
