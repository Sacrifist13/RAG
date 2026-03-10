import sys
import json
import bm25s
from pathlib import Path
from typing import List
from .reader import MinimalSource


class Retriever:

    RED = "\033[91m"
    BOLD = "\033[1m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def __init__(self) -> None:
        chunks_path = Path("data/processed/chunks/sources.json")
        index_dir = Path("data/processed/bm25_index")

        if not index_dir.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Index directory not found: "
                f"{index_dir}{self.RESET}\n\n",
                f"{self.YELLOW} - Run: index command to create index "
                f"directory and chunks datas{self.RESET}",
                file=sys.stderr,
            )
            self.sources = None
            return

        if not chunks_path.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Chunks file not found: "
                f"{chunks_path}{self.RESET}\n\n",
                f"{self.YELLOW} - Run: index command to create index "
                f"directory and chunks datas{self.RESET}",
                file=sys.stderr,
            )
            self.sources = None
            return

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                raw_dicts = json.load(f)

            self.sources = [MinimalSource(**c) for c in raw_dicts]
            self.retriever = bm25s.BM25.load(index_dir, load_corpus=False)

        except Exception:
            print(
                f"{self.RED}{self.BOLD}❌ [ERROR] Command 'search': "
                f"{self.RESET}\n\n",
                f"{self.YELLOW} - Check index and chunks dir{self.RESET}",
                file=sys.stderr,
            )
            self.sources = None

    def retrieve(self, query: str, k: int) -> List[MinimalSource] | None:
        if self.sources is None:
            return None

        query_tokens = bm25s.tokenize(query, stopwords="en")
        documents = self.retriever.retrieve(
            query_tokens, corpus=self.sources, k=k
        )[0]

        return list(documents[0])
