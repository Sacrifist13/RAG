import uuid
from pydantic import BaseModel, Field
from typing import List


class MinimalSource(BaseModel):
    file_path: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)
    content: str = Field(min_length=1, default="")


class MinimalSearchResults(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(min_length=1)
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults] = Field(min_length=1)
    k: int = Field(ge=1)


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
