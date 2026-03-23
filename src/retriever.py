import sys
import bm25s
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Any
from .reader import MinimalSource


class Retriever:
    """Loads BM25 and ChromaDB indexes for hybrid document retrieval."""

    RED = "\033[91m"
    BOLD = "\033[1m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def __init__(self) -> None:
        """
        Initialize BM25 and ChromaDB indexes.

        Sets self.sources to None if indexes are missing or loading fails.
        Run the index command first to create the required indexes.
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

    def _rrf_fusion(
        self,
        bm25_results: List[Any],
        chroma_results: List[Any],
        k: int,
        rrf_k: int,
    ) -> List[MinimalSource]:
        """
        Merge BM25 and ChromaDB results using Reciprocal Rank Fusion.

        Args:
            bm25_results: Ranked results from BM25.
            chroma_results: Ranked results from ChromaDB.
            k: Number of top results to return.
            rrf_k: RRF constant controlling rank smoothing (default 60).

        Returns:
            Top-k sources ranked by combined RRF score.
        """

        scores: dict = {}
        index_map: dict = {}

        for rank, source in enumerate(bm25_results):
            key = f"{source.file_path}_{source.first_character_index}"
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            index_map[key] = source

        for rank, source in enumerate(chroma_results):
            key = f"{source.file_path}_{source.first_character_index}"
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            index_map[key] = source

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)

        return [index_map[key] for key in sorted_keys[:k]]

    def retrieve(self, query: str, k: int) -> List[MinimalSource] | None:
        """
        Retrieve top-k sources using hybrid BM25 and semantic search.

        Args:
            query: The search query string.
            k: Number of top results to return.

        Returns:
            Top-k MinimalSource objects or None if indexes not loaded.
        """
        if self.sources is None:
            return None

        query_tokens = bm25s.tokenize(query, stopwords="en")
        bm25_docs = self.retriever.retrieve(
            query_tokens, corpus=self.sources, k=k
        )[0][0]

        bm25_results = list(bm25_docs)

        chroma_raw = self.collection.query(
            query_texts=[query],
            n_results=k,
        )

        chroma_results = [
            MinimalSource(
                file_path=meta["file_path"],
                first_character_index=meta["first_character_index"],
                last_character_index=meta["last_character_index"],
                content=doc,
            )
            for meta, doc in zip(
                chroma_raw["metadatas"][0],
                chroma_raw["documents"][0],
            )
        ]

        return self._rrf_fusion(bm25_results, chroma_results, k, 60)
