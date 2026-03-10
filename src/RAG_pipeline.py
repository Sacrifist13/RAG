import sys
from pathlib import Path
from .reader import Reader
from .indexer import Indexer
from .retriever import Retriever


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

        for source in best_sources:
            print(f"\n\n{source.content}\n")
