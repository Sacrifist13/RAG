import sys
from typing import List
from transformers import pipeline
from .reader import MinimalSource


class Generator:
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self, model_name="Qwen/Qwen3-0.6B") -> None:
        try:
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                dtype="auto",
                device_map="auto",
            )

        except Exception:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Loading llm model: "
                f"{model_name}{self.RESET}\n",
                file=sys.stderr,
            )
            self.generator = None

    def generate(self, query: str, sources: List[str]) -> str:

        if self.generator is None:
            return None

        context = "\n\n".join(
            [
                f"--- Document {i+1} ---\n{s.content}"
                for i, s in enumerate(sources)
                if isinstance(s, MinimalSource)
            ]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding assistant. Answer the user's"
                    " question based ONLY on the provided documents. If the"
                    " answer is not in the documents, just say you don't know."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context documents:\n{context}\n\nQuestion: {query}"
                ),
            },
        ]

        output_messages = self.generator(
            messages, max_new_tokens=1252, max_length=None, do_sample=False
        )[0]["generated_text"]

        try:
            raw_answer = output_messages[-1]["content"]

            if "</think>" in raw_answer:
                final_answer = raw_answer.split("</think>")[-1].strip()
            else:
                final_answer = raw_answer.strip()

            return final_answer

        except Exception:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] generated answer check "
                f"connection and retry{self.RESET}\n",
                file=sys.stderr,
            )
            return None
