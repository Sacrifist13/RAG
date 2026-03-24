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
    """
    RAGPipeline provides methods for indexing, searching, answering, and
    evaluating using a Retrieval-Augmented Generation (RAG) workflow.

    Methods
    -------
    index(max_chunk_size: int = 2000) -> None
        Indexes raw data for retrieval.

    search(query: str, k: int = 10) -> None
        Retrieves top-k sources for a given query.

    search_dataset(dataset_path: str, k: int = 10,
                   save_directory: str = "data/output/search_results") -> None
        Searches a dataset of questions and saves results.

    answer(query: str, k: int = 10) -> None
        Generates an answer for a query using retrieved sources.

    answer_dataset(student_search_results_path: str,
                   save_directory: str =
                   "data/output/search_results_and_answer"
                   ) -> None
        Generates answers for a dataset of search results.

    evaluate(student_answer_path: str, dataset_path: str, k: int = 10,
             max_context_length: int = 2000) -> None
        Evaluates generated answers against a reference dataset.
    """

    RED = "\033[91m"
    BOLD = "\033[1m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def __init__(self) -> None:
        self.valid_model: List[str] = [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ]

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Indexes and saves processed data from the raw data directory.

        Args:
            max_chunk_size (int): Maximum size of data chunks to process.

        Returns:
            None
        """

        repo_path = Path("data/raw")

        if not repo_path.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Datas directory not found: "
                f"{repo_path}{self.RESET}\n",
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
        """
        Searches for relevant sources given a query.

        Args:
            query (str): The search query.
            k (int, optional): Number of top sources to retrieve. Defaults to
            10.

        Returns:
            None
        """
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
        dataset_path: str = (
            "datasets_public/public/"
            "UnansweredQuestions/dataset_code_public.json"
        ),
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """
        Searches a dataset for relevant sources for each question.

        Args:
            dataset_path (str): Path to the input JSON dataset file.
            k (int, optional): Number of sources to retrieve per question.
            Default is 10.
            save_directory (str, optional): Directory to save results.
            Default is "data/output/search_results".

        Returns:
            None
        """

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
                f"(expected directory): {save_path}{self.RESET}\n",
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

    def answer(
        self, query: str, k: int = 10, model: str = "Qwen/Qwen3-0.6B"
    ) -> None:
        """
        Answers a query using retrieved sources and a language model.

        Args:
            query (str): The input question to answer.
            k (int, optional): Number of sources to retrieve. Defaults to 10.

        Returns:
            None
        """
        retriever = Retriever()

        best_sources = retriever.retrieve(query, k)

        if not best_sources:
            return

        if model not in self.valid_model:
            supported_list = "\n".join([f"  - {m}" for m in self.valid_model])

            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Model name unknown: "
                f"'{model}'{self.RESET}\nAvailable models:{self.RESET}\n"
                f"{self.YELLOW}{supported_list}{self.RESET}\n",
                file=sys.stderr,
            )
            return

        llm = Generator(model_name=model)
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
        model: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """
        Answers a dataset using LLM and saves results to a directory.

        Args:
            student_search_results_path (str): Path to input JSON file.
            save_directory (str): Directory to save output JSON.

        Returns:
            None
        """
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

            search_results_answers: List[MinimalAnswer] | None = []

            if model not in self.valid_model:
                supported_list = "\n".join(
                    [f"  - {m}" for m in self.valid_model]
                )

                print(
                    f"\n{self.RED}{self.BOLD}❌ [ERROR] Model name unknown: "
                    f"'{model}'{self.RESET}\nAvailable models:{self.RESET}\n"
                    f"{self.YELLOW}{supported_list}{self.RESET}\n",
                    file=sys.stderr,
                )
                return

            llm = Generator(model_name=model)

            search_results_answers = llm.generate_batch(
                search_results_datas.search_results
            )

            if not search_results_answers:
                return

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
            return

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
        bonus: bool = False,
    ) -> None:
        """
        Evaluates student answers using the Evaluator class.

        Args:
            student_answer_path (str): Path to student answers file.
            dataset_path (str): Path to dataset file.
            k (int, optional): Top-k results to consider. Defaults to 10.
            max_context_length (int, optional): Max context length.
            Defaults to 2000.

        Returns:
            None
        """
        evaluator = Evaluator(
            student_answer_path, dataset_path, max_context_length
        )

        if not evaluator.compared:
            return

        evaluator.evaluate(k, bonus)
