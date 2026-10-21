from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description="Choose kb for HR policies or employee-specific questions; "
                    "direct for greetings and unrelated general questions."
    )
    reason: str = Field(
        default="",
        description="Brief explanation for the routing decision.",
    )


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="Whether the evidence directly and sufficiently answers "
                    "the user's question."
    )
    reason: str = Field(
        default="",
        description="Brief explanation for the evidence grade.",
    )


class AgentState(TypedDict):
    question: str             # actual question asked
    current_query: str        # query after rewriting
    kb_docs: list[Document]   # retrieved private KB chunks
    web_results: str          # web search evidence (fallback)
    kb_grade: str             # LLM judgement of KB evidence
    web_grade: str            # LLM judgement of web evidence
    answer: str               # final answer
    source_used: str          # where the answer came from (kb, web, direct, ...)
    retry_count: int          # number of query rewrites so far
    trace: list[str]          # execution steps shown in the UI
    citations: list[dict]     # sources used for the answer