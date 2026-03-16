import sys
import torch
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from .models import MinimalSource, MinimalSearchResults, MinimalAnswer


class Generator:
    """
    Generator class for LLM-based answer generation from provided sources.

    Args:
        model_name (str): Name of the pretrained LLM model.

    Attributes:
        tokenizer: Tokenizer for the LLM.
        model: LLM model instance.

    Methods:
        generate_message(question, sources):
            Build chat messages from question and sources.
            Args:
                question (str): User's question.
                sources (List[MinimalSource]): Source docs.
            Returns:
                List[Dict[str, str]]: Chat messages.

        generate(query, sources):
            Generate answer for a single query.
            Args:
                query (str): User's question.
                sources (List[MinimalSource]): Source docs.
            Returns:
                str: Generated answer.

        generate_batch(datas):
            Generate answers for a batch of queries.
            Args:
                datas (List[MinimalSearchResults]): Batch input.
            Returns:
                List[MinimalAnswer] | None: Answers or None if model not
                loaded.
    """

    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """
        Initializes tokenizer and model for causal language modeling.

        Args:
            model_name (str): Name or path of the pretrained model.

        Returns:
            None
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left"
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

        except Exception as e:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Loading llm model: "
                f"{model_name} - {e}{self.RESET}\n",
                file=sys.stderr,
            )
            self.model = None
            self.tokenizer = None

    def generate_message(
        self, question: str, sources: List[MinimalSource]
    ) -> List[Dict[str, str]]:
        """
        Generate chat messages for answering a question using given sources.

        Args:
            question (str): The user's question.
            sources (List[MinimalSource]): List of source documents.

        Returns:
            List[Dict[str, str]]: Chat messages for the assistant.
        """
        context = "\n\n".join(
            [
                f"--- Document {i+1} ---\n{s.content[:500]}"
                for i, s in enumerate(sources[:3])
                if isinstance(s, MinimalSource)
            ]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding assistant. Answer the user's"
                    " question based ONLY on the provided documents."
                    "/no_think"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context documents:\n{context}\n\nQuestion: {question}"
                ),
            },
        ]

        return messages

    def generate(self, query: str, sources: List[MinimalSource]) -> str:
        """
        Generate a response to a query using provided sources.

        Args:
            query (str): The input question or prompt.
            sources (List[MinimalSource]): List of context sources.

        Returns:
            str: Generated answer or empty string on failure.
        """

        if self.model is None:
            return ""

        message = self.generate_message(query, sources)

        try:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][input_length:]
            raw_answer = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            if "</think>" in raw_answer:
                return str(raw_answer.split("</think>")[-1].strip())
            return str(raw_answer.strip())

        except Exception as e:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] {e}{self.RESET}\n",
                file=sys.stderr,
            )
            return ""

    def generate_batch(
        self,
        datas: List[MinimalSearchResults],
    ) -> List[MinimalAnswer] | None:
        """
        Generate answers for a batch of search results.

        Args:
            datas (List[MinimalSearchResults]): Batch of search results.

        Returns:
            List[MinimalAnswer] | None: Generated answers or None if no model.
        """
        if not self.model:
            return None

        all_answers: List[MinimalAnswer] = []

        for i in tqdm(range(0, len(datas), 2), desc="Processing answers"):
            chunk = datas[i: i + 2]

            try:
                texts = [
                    self.tokenizer.apply_chat_template(
                        self.generate_message(
                            data.question, data.retrieved_sources
                        ),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for data in chunk
                    if isinstance(data, MinimalSearchResults)
                ]

                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                input_length = inputs["input_ids"].shape[1]

                for data, output in zip(chunk, output_ids):
                    new_tokens = output[input_length:]
                    raw = self.tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    )

                    if "</think>" in raw:
                        raw = raw.split("</think>")[-1].strip()

                    all_answers.append(
                        MinimalAnswer(
                            question_id=data.question_id,
                            question=data.question,
                            retrieved_sources=data.retrieved_sources,
                            answer=raw.strip(),
                        )
                    )

            except Exception as e:
                print(
                    f"\n{self.RED}{self.BOLD}❌ [ERROR] Batch {i//2 + 1}: "
                    f"{e}{self.RESET}\n",
                    file=sys.stderr,
                )
                for data in chunk:
                    all_answers.append(
                        MinimalAnswer(
                            question_id=data.question_id,
                            question=data.question,
                            retrieved_sources=data.retrieved_sources,
                            answer="",
                        )
                    )

        return all_answers
