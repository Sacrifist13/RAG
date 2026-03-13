import sys
from typing import List
from transformers import pipeline
from tqdm import tqdm
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

    def generate(self, query: str, sources: List[MinimalSource]) -> str:

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
        try:
            output_messages = self.generator(
                messages, max_new_tokens=1024, max_length=None, do_sample=False
            )[0]["generated_text"]

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

    def generate_batch(
        self, questions: List[str], sources: List[List[MinimalSource]]
    ) -> List[str]:

        if self.generator is None:
            return [""] * len(questions)

        all_messages = []

        for question, source in zip(questions, sources):
            context = "\n\n".join(
                [
                    f"--- Document {i+1} ---\n{s.content}"
                    for i, s in enumerate(source)
                    if isinstance(s, MinimalSource)
                ]
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful coding assistant. Answer the user's"
                        " question based ONLY on the provided documents. "
                        "Be concise. If "
                        "the answer is not in the documents, just say you "
                        "don't know."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context documents:\n{context}\n\nQuestion: "
                        f"{question}"
                    ),
                },
            ]

            all_messages.append(messages)

        if self.generator.tokenizer.pad_token_id is None:
            self.generator.tokenizer.pad_token_id = (
                self.generator.tokenizer.eos_token_id
            )

        self.generator.tokenizer.padding_side = "left"

        final_answers = []
        batch_size = 16

        try:

            for i in tqdm(
                range(0, len(all_messages), batch_size),
                desc="Generating AI answers",
            ):
                chunk = all_messages[i : i + batch_size]
                batch_outputs = self.generator(
                    chunk,
                    max_new_tokens=64,
                    max_length=None,
                    do_sample=False,
                    batch_size=batch_size,
                )

                for output in batch_outputs:
                    raw_answer = output[0]["generated_text"][-1]["content"]
                    if "</think>" in raw_answer:
                        final_answer = raw_answer.split("</think>")[-1].strip()
                    else:
                        final_answer = raw_answer.strip()

                    final_answers.append(final_answer)

            return final_answers

        except Exception as e:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Batch generation failed: "
                f"{e}{self.RESET}\n",
                file=sys.stderr,
            )
            return [""] * len(questions)
