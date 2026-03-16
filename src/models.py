import uuid
from pydantic import BaseModel, Field
from typing import List


class MinimalSource(BaseModel):
    """
    MinimalSource model for text source metadata.

    Args:
        file_path (str): Path to the source file.
        first_character_index (int): Start index of the content.
        last_character_index (int): End index of the content.
        content (str): Extracted text content.

    Returns:
        MinimalSource: Instance with file and content info.
    """
    file_path: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)
    content: str = Field(min_length=1, default="")


class MinimalSearchResults(BaseModel):
    """
    Minimal search result model.

    Args:
        question_id (str): Unique question identifier.
        question (str): The question text.
        retrieved_sources (List[MinimalSource]): Sources retrieved.

    Returns:
        MinimalSearchResults: Model instance with search results.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(min_length=1)
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    MinimalAnswer extends MinimalSearchResults with an answer field.

    Args:
        answer (str): The answer string.

    Returns:
        MinimalAnswer: An object with search results and an answer.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    Represents search results for students.

    Args:
        search_results (List[MinimalSearchResults]): List of search results.
        k (int): Number of top results to return.

    Returns:
        StudentSearchResults: Model with search results and k value.
    """
    search_results: List[MinimalSearchResults] = Field(min_length=1)
    k: int = Field(ge=1)


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """
    Extends StudentSearchResults with a list of MinimalAnswer results.

    Args:
        search_results (List[MinimalAnswer]): List of answer results.

    Returns:
        StudentSearchResultsAndAnswer: Instance with search results.
    """
    search_results: List[MinimalAnswer]  # type: ignore[assignment]


class UnansweredQuestion(BaseModel):
    """
    Represents a question that has not been answered.

    Args:
        question_id (str): Unique ID for the question.
        question (str): The question text.

    Returns:
        UnansweredQuestion: Instance with question details.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    AnsweredQuestion extends UnansweredQuestion with answer and sources.

    Args:
        sources (List[MinimalSource]): List of source documents.
        answer (str): The provided answer.

    Returns:
        AnsweredQuestion: An answered question with sources.
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    RagDataset holds a list of answered or unanswered questions.

    Args:
        rag_questions (List[AnsweredQuestion | UnansweredQuestion]):
            List of questions, answered or not.

    Returns:
        RagDataset: Instance containing the questions.
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
