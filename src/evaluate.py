import sys
import json
from pathlib import Path
from typing import Dict, List
from .models import (
    StudentSearchResults,
    RagDataset,
    AnsweredQuestion,
    MinimalSource,
)


class Evaluator:
    """
    Evaluator for validating and scoring student answers.

    Args:
        student_answer_path (str): Path to student answers file.
        dataset_path (str): Path to reference dataset file.
        max_context_length (int): Max allowed context length.

    Methods:
        evaluate(k: int): Evaluate recall@k for student answers.
    """

    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(
        self,
        student_answer_path: str,
        dataset_path: str,
        max_context_length: int,
    ) -> None:
        """
        Initializes the evaluation object by validating and loading student
        answers and dataset files.

        Args:
            student_answer_path (str): Path to the student answers JSON file.
            dataset_path (str): Path to the dataset JSON file.
            max_context_length (int): Maximum allowed length for source
            content.

        Returns:
            None
        """
        self.compared: Dict[str, Dict[str, List[MinimalSource]]] | None = {}

        valid: bool = True
        no_valid: int = 0

        student_path = Path(student_answer_path)
        data_path = Path(dataset_path)

        if not student_path.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Student answers file not "
                f"found: {student_path}{self.RESET}\n",
                file=sys.stderr,
            )
            self.compared = None
            return

        if not student_path.is_file():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Student answers need to be "
                f"a file, found: {student_path}{self.RESET}\n",
                file=sys.stderr,
            )
            self.compared = None
            return

        if not data_path.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Dataset file not found: "
                f"found: {data_path}{self.RESET}\n",
                file=sys.stderr,
            )
            self.compared = None
            return

        if not data_path.is_file():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Dataset answers need to be "
                f"a file, found: {data_path}{self.RESET}\n",
                file=sys.stderr,
            )
            self.compared = None
            return

        try:
            with open(student_path, "r", encoding="utf-8") as f:
                student_json = json.load(f)
                student_answers = StudentSearchResults(**student_json)

                for search_results in student_answers.search_results:
                    for sources in search_results.retrieved_sources:
                        if len(sources.content) > max_context_length:
                            valid = False
                            no_valid += 1
                if not valid:
                    print(
                        f"Student data is valid: False ({no_valid} non valid "
                        "datas)"
                    )
                else:
                    print("Student data is valid: True")

            with open(data_path, "r", encoding="utf-8") as f:
                data_json = json.load(f)
                data_answers = RagDataset(**data_json)
                print(
                    "Total number of questions: "
                    f"{len(data_answers.rag_questions)}"
                )
                answered = {
                    q.question_id: q.sources
                    for q in data_answers.rag_questions
                    if isinstance(q, AnsweredQuestion)
                }
                print(
                    f"Total number of questions with sources: {len(answered)}"
                )

            self.compared = {
                result.question_id: {
                    "student": result.retrieved_sources,
                    "data": answered.get(result.question_id, []),
                }
                for result in student_answers.search_results
                if result.question_id in answered
            }

            print(
                "Total number of questions with student sources: "
                f"{len(self.compared)}\n"
            )

        except Exception as e:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] {e}{self.RESET}\n",
                file=sys.stderr,
            )
            self.compared = None
            return

    def _check_overlap(
        self, r_start: int, r_end: int, c_start: int, c_end: int
    ) -> float:
        """
        Calculate the proportion of overlap between two ranges.

        Args:
            r_start (int): Start of the reference range.
            r_end (int): End of the reference range.
            c_start (int): Start of the candidate range.
            c_end (int): End of the candidate range.

        Returns:
            float: Fraction of the candidate range overlapped by the reference
            range.
        """
        overlap_start = max(r_start, c_start)
        overlap_end = min(r_end, c_end)
        overlap_length = max(0, overlap_end - overlap_start)
        correct_length = c_end - c_start

        return overlap_length / correct_length

    def _check_sources(
        self, data_source: MinimalSource, find_sources: List[MinimalSource]
    ) -> bool:
        """
        Checks if the given data_source overlaps with any source in
        find_sources by at least 5%.

        Args:
            data_source (MinimalSource): The source to check.
            find_sources (List[MinimalSource]): List of sources to compare
            against.

        Returns:
            bool: True if overlap >= 5% with any source, False otherwise.
        """
        for f in find_sources:
            if f.file_path == data_source.file_path:
                overlap = self._check_overlap(
                    data_source.first_character_index,
                    data_source.last_character_index,
                    f.first_character_index,
                    f.last_character_index,
                )
                if overlap >= 0.05:
                    return True
        return False

    def _recall(
        self,
        data_sources: List[MinimalSource],
        student_sources: List[MinimalSource],
    ) -> float:
        """
        Calculate recall score for student sources.

        Args:
            data_sources (List[MinimalSource]): Reference sources.
            student_sources (List[MinimalSource]): Sources to evaluate.

        Returns:
            float: Recall score (0.0-1.0).
        """
        if not data_sources:
            return 1

        found = 0
        for data in data_sources:
            if self._check_sources(data, student_sources):
                found += 1

        return found / len(data_sources)

    def evaluate(self, k: int, bonus: bool) -> None:
        """
        Evaluates recall and optionally precision at various cutoffs for
        compared questions.

        Args:
            k (int): Maximum cutoff value for evaluation metrics
            (e.g., Recall@k, Precision@k).
            bonus (bool): If True, calculates and prints Precision@k alongside
            recall.

        Returns:
            None
        """

        if not self.compared:
            return

        cutoff = [1, 3, 5, k]
        print("Evaluation Results")
        print("========================================")
        print(f"Questions evaluated: {len(self.compared.keys())}")
        for c in cutoff:
            scores = []

            for q in self.compared.keys():
                student_sources: List[MinimalSource] = self.compared[q][
                    "student"
                ][:c]
                data_sources: List[MinimalSource] = self.compared[q]["data"]
                scores.append(self._recall(data_sources, student_sources))
            mean_recall = sum(scores) / len(scores)
            print(f"Recall@{c}: {mean_recall:.3f}")

        if bonus:
            for c in [1, 3, 5, k]:
                precisions = []
                for q in self.compared.keys():
                    student_top_k = self.compared[q]["student"][:c]
                    data_sources = self.compared[q]["data"]

                    found = sum(
                        1
                        for s in student_top_k
                        if self._check_sources(s, data_sources)
                    )
                    precisions.append(found / c)

                print(
                    f"Precision@{c}: {sum(precisions) / len(precisions):.3f}"
                )
