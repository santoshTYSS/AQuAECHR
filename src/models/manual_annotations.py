from typing import Literal
from pydantic import BaseModel


class Citation(BaseModel):
    case_id: str
    case_name: str
    paragraph_number: int
    paragraph_text: str


class SentenceWithCitations(BaseModel):
    sentence: str
    citations: list[Citation]


class Generation(BaseModel):
    experiment: str
    answer: str
    sentences_with_citations: list[SentenceWithCitations] | None = None


class Annotation(BaseModel):
    claim_order: list[str]
    citation_faithfulness_order: list[str]
    citation_similarity_order: list[str]
    better_than_target: list[str]
    note: str | None = None


class TargetCitation(BaseModel):
    case_name: str
    case_id: str
    paragraph_numbers: list[int]
    paragraphs_map: dict[str, str]


class EvaluationBatch(BaseModel):
    axis: Literal["model", "approach"]
    question_number: int
    question: str
    answer: str
    citations: list[TargetCitation]
    generations: list[Generation]
    annotation: Annotation | None = None
