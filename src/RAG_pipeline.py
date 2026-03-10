import sys
import uuid
import json
from tqdm import tqdm
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
    YELLOW = "\033[93m"
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

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:

        data_path = Path(dataset_path)
        save_path = Path(save_directory)

        if not data_path.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Datas file not found: "
                f"{data_path}{self.RESET}\n",
                file=sys.stderr,
            )
            return

        if not data_path.is_file():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Datas file is a directory "
                f"(expected json file): {data_path}{self.RESET}\n",
                file=sys.stderr,
            )
            return

        if save_path.exists() and save_path.is_file():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Save directory is a file "
                f"(expected directory): {save_path.name}{self.RESET}\n",
                file=sys.stderr,
            )
            return

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

        except Exception:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Datas file wrong format "
                f"(.json required): {data_path}{self.RESET}\n",
                file=sys.stderr,
            )
            return

        if "rag_questions" not in json_data:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Invalid dataset format: "
                f"{data_path}{self.RESET}\n\n"
                f"{self.YELLOW} - Expected a JSON object containing the root "
                f"key 'rag_questions'.{self.RESET}",
                file=sys.stderr,
            )
            return

        retriever = Retriever()

        if not retriever.sources:
            return

        search_results: List[MinimalSearchResults] = []

        try:
            for question in tqdm(
                json_data["rag_questions"], desc="Processing questions"
            ):
                best_sources = retriever.retrieve(question["question"], k)

                if not best_sources:
                    continue

                search_results.append(
                    MinimalSearchResults(
                        question_id=question["question_id"],
                        question=question["question"],
                        retrieved_sources=best_sources,
                    )
                )
        except Exception:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Searching results error"
                f"{data_path}{self.RESET}",
                file=sys.stderr,
            )
            return

        student_search_results = StudentSearchResults(
            search_results=search_results, k=k
        )

        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / data_path.name

        with open(file_path, "w", encoding="utf-8") as f:
            json_string = student_search_results.model_dump_json(indent=4)
            f.write(json_string)

        print(f"Saved student_search_results to {file_path}")
        return
