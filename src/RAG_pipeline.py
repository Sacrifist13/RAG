import sys
from pathlib import Path
from .reader import Reader
from .indexer import Indexer


class RAGPipeline:

    RED = "\033[91m"
    YELLOW = "\033[93m"
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

        print(f"\n-- STARTING INDEXING {repo_path} directory --\n")

        reader = Reader()
        datas_formated = reader.read_split(repo_path, max_chunk_size)

        if not datas_formated:
            return

        indexer = Indexer()
        indexer.index_save(datas_formated)
