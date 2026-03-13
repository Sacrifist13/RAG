import sys
from pathlib import Path


class Evaluator:
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self, student_answer_path: str, dataset_path: str) -> None:
        student_path = Path(student_answer_path)
        data_path = Path(dataset_path)

        if not student_path.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Student answers file not "
                f"found: {student_path}{self.RESET}\n",
                file=sys.stderr,
            )

        if not student_path.is_file():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Student answers need to be "
                f"a file, found: {student_path}{self.RESET}\n",
                file=sys.stderr,
            )

        if not data_path.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Student answers file not "
                f"found: {data_path}{self.RESET}\n",
                file=sys.stderr,
            )

        if not data_path.is_file():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Student answers need to be "
                f"a file, found: {data_path}{self.RESET}\n",
                file=sys.stderr,
            )

        
