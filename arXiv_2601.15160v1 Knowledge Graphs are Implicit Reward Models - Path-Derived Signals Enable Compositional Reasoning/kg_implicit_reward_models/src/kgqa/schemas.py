from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict


class QASpec(BaseModel):
    """
    Structured spec derived from a reasoning path, graph-agnostic.
    """

    model_config = ConfigDict(extra="forbid")

    start_node_id: str
    answer_node_id: str

    # For interpretability / LLM prompting
    start_text: str
    answer_text: str

    relations: List[str] = Field(default_factory=list)
    intermediate_node_ids: List[str] = Field(default_factory=list)
    intermediate_texts: List[str] = Field(default_factory=list)

    # Triples for the ground-truth reasoning chain
    ground_truth_path: List[Tuple[str, str, str]]  # (src_id, relation, dst_id)

    hop_length: int = Field(ge=1)


class MCQItem(BaseModel):
    """
    Final QA sample to write to JSONL (or return from pipeline).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    difficulty: int = Field(ge=1)
    hop_length: int = Field(ge=1)

    question: str
    options: Dict[str, str]  # {"A": "...", "B": "...", ...}
    correct_option: Literal["A", "B", "C", "D"]

    # Supervision signals
    spec: QASpec


class MCQGenerationResult(BaseModel):
    """
    Optional wrapper if you want status + errors.
    """

    model_config = ConfigDict(extra="forbid")

    item: Optional[MCQItem] = None
    error: Optional[str] = None
