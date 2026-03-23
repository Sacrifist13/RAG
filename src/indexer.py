import bm25s
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
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

    def __init__(self) -> None:
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
