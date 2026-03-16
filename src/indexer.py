import bm25s
import json
from pathlib import Path
from typing import List
from .reader import MinimalSource


class Indexer:
    """
    Indexer class for saving and indexing document sources.

    Methods
    -------
    index_save(sources: List[MinimalSource]) -> None
        Save sources as JSON and build BM25 index.

    Args:
        sources (List[MinimalSource]): List of source objects to index.

    Returns:
        None
    """
    def index_save(self, sources: List[MinimalSource]) -> None:
        chunks_dir = Path("data/processed/chunks")
        index_dir = Path("data/processed/bm25_index")

        contents = []
        chunks = []

        for s in sources:
            contents.append(s.content)
            chunks.append(s.model_dump(mode="json"))

        chunks_dir.mkdir(parents=True, exist_ok=True)
        file_path = chunks_dir / "sources.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4)

        contents_tokens = bm25s.tokenize(contents, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(contents_tokens)
        retriever.save(index_dir)
