import sys
import bm25s
import json
from pathlib import Path
from typing import List
from .reader import MinimalSource


class Retriever:

    RED = "\033[91m"
    BOLD = "\033[1m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def retrieve(self, query: str, k: int) -> List[MinimalSource] | None:

        index_dir = Path("data/processed/bm25_index")
        chunks_path = Path("data/processed/chunks/sources.json")

        if not index_dir.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Index directory not found: "
                f"{index_dir}{self.RESET}\n\n",
                f"{self.YELLOW} - Run: index command to create index "
                f"directory and chunks datas{self.RESET}",
                file=sys.stderr,
            )
            return None

        if not chunks_path.exists():
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Chunks file not found: "
                f"{chunks_path}{self.RESET}\n\n",
                f"{self.YELLOW} - Run: index command to create index "
                f"directory and chunks datas{self.RESET}",
                file=sys.stderr,
            )
            return None

        try:
            retriever = bm25s.BM25.load(index_dir, load_corpus=False)

            with open(chunks_path, "r", encoding="utf-8") as f:
                raw_dicts = json.load(f)

            sources = [MinimalSource(**c) for c in raw_dicts]
            query_tokens = bm25s.tokenize(query, stopwords="en")
            documents = retriever.retrieve(query_tokens, corpus=sources, k=k)[
                0
            ]

            return list(documents[0])

        except Exception:
            print(
                f"{self.RED}{self.BOLD}❌ [ERROR] Command 'search': "
                f"{self.RESET}\n\n",
                f"{self.YELLOW} - Check index and chunks dir{self.RESET}",
                file=sys.stderr,
            )
            return None
