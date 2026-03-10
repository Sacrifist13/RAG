import sys
import uuid
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from .reader import Reader, MinimalSource
from .indexer import Indexer
from .retriever import Retriever


class MinimalSearchResults(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(min_length=1)
    retrieved_sources: List[MinimalSource]


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults] = Field(min_length=1)
    k: int = Field(ge=1)


class RAGPipeline:

    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def index(self, max_chunk_size: int = 2000) -> None:

        repo_path = Path("data/raw")

        if not repo_path.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Datas directory not found: "
                f"{repo_path.name}{self.RESET}\n",
                file=sys.stderr,
            )
            return

        reader = Reader()
        datas_formated = reader.read_split(repo_path, max_chunk_size)

        if not datas_formated:
            return

        indexer = Indexer()
        indexer.index_save(datas_formated)

        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10) -> None:
        retriever = Retriever()

        best_sources = retriever.retrieve(query, k)

        if not best_sources:
            return

        search_result = MinimalSearchResults(
            question=query, retrieved_sources=best_sources
        )

        final_output = StudentSearchResults(
            search_results=[search_result], k=k
        )

        print(final_output.model_dump_json(indent=4))
