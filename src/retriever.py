import sys
import json
import bm25s
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List
from .reader import MinimalSource


class Retriever:
    """
    Retriever loads sources and BM25 index for document retrieval.

    Attributes:
        sources (list[MinimalSource] | None): Loaded document sources.
        retriever (bm25s.BM25): BM25 retriever instance.

    Methods:
        retrieve(query: str, k: int) -> list[MinimalSource] | None:
            Retrieve top-k relevant sources for a given query.

    Args:
        query (str): Query string to search for.
        k (int): Number of top results to return.

    Returns:
        list[MinimalSource] | None: Top-k sources or None if not loaded.
    """

    RED = "\033[91m"
    BOLD = "\033[1m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def __init__(self) -> None:
        """
        Initializes the retriever by loading sources and BM25 index.

        Raises:
            Prints error and sets self.sources to None if files are missing or
            loading fails.
        """
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
        try:
            self.retriever = bm25s.BM25.load(index_dir, load_corpus=False)

            self.client = chromadb.PersistentClient(
                path="data/processed/chroma_index"
            )
            self.collection = self.client.get_collection(
                name="chunks",
                embedding_function=(
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="paraphrase-MiniLM-L3-v2"
                    )
                ),
            )

            results = self.collection.get()

            self.sources = [
                MinimalSource(
                    file_path=meta["file_path"],
                    first_character_index=meta["first_character_index"],
                    last_character_index=meta["last_character_index"],
                    content=doc,
                )
                for meta, doc in zip(
                    results["metadatas"], results["documents"]
                )
            ]

        except Exception as e:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] {e}{self.RESET}\n",
                file=sys.stderr,
            )
            self.sources = None
            return

    def retrieve(self, query: str, k: int) -> List[MinimalSource] | None:
        """
        Retrieve top-k relevant MinimalSource objects for a given query.

        Args:
            query (str): The search query.
            k (int): Number of top results to return.

        Returns:
            list[MinimalSource] | None: Top-k sources or None if no sources.
        """
        if self.sources is None:
            return None

        query_tokens = bm25s.tokenize(query, stopwords="en")
        documents = self.retriever.retrieve(
            query_tokens, corpus=self.sources, k=k
        )[0]

        return list(documents[0])
