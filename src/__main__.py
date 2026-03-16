import fire
import sys
from .RAG_pipeline import RAGPipeline

if __name__ == "__main__":
    try:
        fire.Fire(RAGPipeline())
    except Exception as e:
        print(
            f"\n❌ [ERROR] {e}",
            file=sys.stderr,
        )