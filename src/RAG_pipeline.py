import sys
import json
from tqdm import tqdm
from pathlib import Path
from typing import List
from .reader import Reader
from .indexer import Indexer
from .retriever import Retriever
from .generator import Generator
from .evaluate import Evaluator
from .models import (
    MinimalSearchResults,
    StudentSearchResults,
    RagDataset,
    UnansweredQuestion,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)


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

            rag_dataset = RagDataset(**json_data)
            questions: List[UnansweredQuestion] = rag_dataset.rag_questions

        except Exception as e:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] {e}{self.RESET}\n",
                file=sys.stderr,
            )
            return

        retriever = Retriever()

        if not retriever.sources:
            return

        search_results: List[MinimalSearchResults] = []

        try:
            for question in tqdm(questions, desc="Processing questions"):
                best_sources = retriever.retrieve(question.question, k)

                if not best_sources:
                    continue

                search_results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=best_sources,
                    )
                )
        except Exception as e:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] {e}{self.RESET}\n",
                file=sys.stderr,
            )
            return None

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

    def answer(self, query: str, k: int = 10) -> None:
        retriever = Retriever()

        best_sources = retriever.retrieve(query, k)

        if not best_sources:
            return

        llm = Generator()
        raw_answer = llm.generate(query, best_sources)

        if not raw_answer:
            return

        answer = MinimalAnswer(
            question=query, retrieved_sources=best_sources, answer=raw_answer
        )

        print(answer.model_dump_json(indent=4))

    def answer_dataset(
        self,
        student_search_results_path: str = (
            "data/output/search_results/dataset_code_public.json"
        ),
        save_directory: str = "data/output/search_results_and_answer",
    ):
        data_path = Path(student_search_results_path)
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
                search_results_datas = StudentSearchResults(**json_data)

            search_results_answers: List[MinimalAnswer] = []
            llm = Generator()

            search_results_answers = llm.generate_batch(
                search_results_datas.search_results
            )

            student_search_results_and_answer = StudentSearchResultsAndAnswer(
                search_results=search_results_answers, k=search_results_datas.k
            )
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / data_path.name

            with open(file_path, "w", encoding="utf-8") as f:
                json_string = (
                    student_search_results_and_answer.model_dump_json(indent=4)
                )
                f.write(json_string)

                print(
                    f"Saved student_search_results_and_answer to {file_path}"
                )
                return

        except Exception as e:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] {e}{self.RESET}\n",
                file=sys.stderr,
            )
            return None

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
    ) -> None:
        evaluator = Evaluator(
            student_answer_path, dataset_path, max_context_length
        )

        if not evaluator.compared:
            return

        evaluator.evaluate(k)
