import fire
import sys
from .RAG_pipeline import RAGPipeline

if __name__ == "__main__":
    """
    Main entry point for running the RAGPipeline CLI with Fire.
    Handles exceptions and prints errors to stderr.
    """
    try:
        fire.Fire(RAGPipeline())
    except Exception as e:
        print(
            f"\n❌ [ERROR] {e}",
            file=sys.stderr,
        )
