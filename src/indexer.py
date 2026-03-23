import bm25s
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
from typing import List
from .reader import MinimalSource


class Indexer:
    """Builds BM25 and ChromaDB indexes from document sources."""

    def __init__(self) -> None:
        """Initialize ChromaDB persistent client and collection."""

        self.client = chromadb.PersistentClient(
            path="data/processed/chroma_index"
        )
        self.collection = self.client.get_or_create_collection(
            name="chunks",
            embedding_function=(
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="paraphrase-MiniLM-L3-v2"
                )
            ),
        )

    def index_save(self, sources: List[MinimalSource]) -> None:
        """
        Index sources into BM25 and ChromaDB.

        Encodes document content using SentenceTransformer and stores
        embeddings with metadata in ChromaDB. Also builds and saves
        a BM25 index for lexical search.

        Args:
            sources: List of MinimalSource objects to index.
        """

        index_dir = Path("data/processed/bm25_index")

        index_dir.mkdir(parents=True, exist_ok=True)

        contents = [s.content for s in sources]

        contents_tokens = bm25s.tokenize(contents, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(contents_tokens)
        retriever.save(index_dir)

        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        embeddings = model.encode(
            contents,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        batch_size = 5000

        for i in tqdm(
            range(0, len(sources), batch_size), desc="Indexing ChromaDB"
        ):
            batch = sources[i: i + batch_size]
            self.collection.add(
                ids=[str(j) for j in range(i, i + len(batch))],
                documents=[s.content for s in batch],
                embeddings=embeddings[i: i + len(batch)].tolist(),
                metadatas=[
                    {
                        "file_path": s.file_path,
                        "first_character_index": s.first_character_index,
                        "last_character_index": s.last_character_index,
                    }
                    for s in batch
                ],
            )
