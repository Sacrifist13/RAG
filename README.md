*This project has been created as part of the 42 curriculum by kkraft.*

<details open>
<summary><h2>📑 Table of Contents</h2></summary>

1. [Description](#description)
2. [System Architecture](#system-architecture)
3. [Component Deep Dive](#component-deep-dive)
4. [Chunking Strategy](#chunking-strategy)
5. [Retrieval Method](#retrieval-method)
6. [Performance Analysis](#performance-analysis)
7. [Design Decisions](#design-decisions)
8. [Challenges Faced](#challenges-faced)
9. [Instructions & Example Usage](#instructions--example-usage)
10. [Resources](#resources)

</details>

---

## Description

This project implements a complete Retrieval-Augmented Generation (RAG) system designed to answer questions about a given codebase, specifically targeting the vLLM repository. The system works by ingesting raw Python and Markdown files, segmenting them into manageable chunks, and indexing them for fast semantic search. When queried, it retrieves the most relevant contexts and utilizes a local Large Language Model (`Qwen/Qwen3-0.6B`) to generate accurate, evidence-based responses. The objective is to combine the reasoning capabilities of an LLM with highly specific, up-to-date external knowledge.

---

## System Architecture

The pipeline follows a highly modular design to separate concerns and ensure maintainability. The main components interact as follows:

1. **Ingestion**: Raw documents are parsed and segmented.
2. **Indexing**: Processed segments are tokenized and stored in a searchable index.
3. **Retrieval**: User queries are matched against the index to find the most relevant document chunks.
4. **Generation**: The retrieved chunks are formatted into a prompt and fed into the local LLM to produce a final answer.
5. **Evaluation**: An integrated evaluator measures the system's retrieval accuracy using Recall@k metrics.

---

## Component Deep Dive

<details>
<summary><b>1. Entry Point: <code>student.py</code> & <code>RAG_pipeline.py</code></b></summary>

* **`student.py`**: Acts as the minimalist Command-Line Interface (CLI). It uses the `fire` library to automatically expose the methods of `RAGPipeline` as terminal commands.
* **`RAG_pipeline.py`**: The central orchestrator.
  * `index(max_chunk_size)`: Coordinates the reading of `data/raw` and saves the processed index to `data/processed/`.
  * `search(query, k)`: Retrieves top `k` sources using the `Retriever` and dumps the JSON representation.
  * `search_dataset(...)`: Processes a batch of unanswered questions from a JSON dataset using `tqdm` for progress tracking.
  * `answer(query, k)`: Chains the `Retriever` and `Generator` to output a single contextualized answer.
  * `answer_dataset(...)`: Generates answers in batches using the LLM for a whole dataset.
  * `evaluate(...)`: Triggers the evaluator to compare student predictions against the ground truth.
</details>

<details>
<summary><b>2. Data Ingestion: <code>reader.py</code></b></summary>

Responsible for parsing the file tree and segmenting files.
* **`read_split()`**: Iterates through the raw data directory (ignoring `.git` and `__pycache__`). It dynamically selects the appropriate `langchain` text splitter based on the file extension (`.py`, `.md`, or generic text) to respect structural boundaries like functions and paragraphs. Outputs an array of `MinimalSource` Pydantic models.
</details>

<details>
<summary><b>3. Indexing & Retrieval: <code>indexer.py</code> & <code>retriever.py</code></b></summary>

* **`indexer.py`**: Extracts text contents, tokenizes them with stopword removal, and feeds them into a `bm25s.BM25` indexer. It saves both the raw chunk data as JSON and the binary BM25 index to the disk.
* **`retriever.py`**: Loads the pre-computed BM25 index and chunk references. The `retrieve(query, k)` method tokenizes the user query, queries the BM25 model, and returns the top `k` matching `MinimalSource` objects.
</details>

<details>
<summary><b>4. AI Generation: <code>generator.py</code></b></summary>

Manages the interaction with the HuggingFace `transformers` ecosystem.
* **Model Loading**: Initializes `Qwen/Qwen3-0.6B` and its tokenizer, forcing `torch.bfloat16` for memory efficiency.
* **`generate_message()`**: Formats a strict system prompt instructing the model to rely *only* on the provided context, appending the top retrieved documents.
* **`generate()` / `generate_batch()`**: Handles token generation, stripping special tokens (like `</think>`), and returning structured `MinimalAnswer` objects.
</details>

<details>
<summary><b>5. Evaluation Model: <code>evaluate.py</code> & <code>models.py</code></b></summary>

* **`models.py`**: Uses Pydantic to strictly type expected JSON outputs (e.g., `MinimalSource`, `StudentSearchResults`, `RagDataset`), ensuring schema validation at runtime.
* **`evaluate.py`**: Validates character constraints (max length). It calculates Recall@k by determining if the predicted text chunk overlaps by at least 5% with the ground truth chunk (`_check_overlap`).
</details>

---

## Chunking Strategy

To efficiently process different file types, the system implements specialized chunking mechanisms using `langchain_text_splitters`.

* **Python Code**: Uses `Language.PYTHON` splitters to prevent breaking classes or functions mid-way.
* **Markdown**: Uses `Language.MARKDOWN` splitters to respect headers and paragraph blocks.
* **Other Text**: Uses a custom regex separator sequence (`\n\n`, `\n`, `.`, ` `).
* **Size**: The maximum chunk size defaults to **2000 characters**, with an overlap of **10%** (200 characters) to ensure contextual continuity across chunk boundaries. This is configurable via the CLI.

---

## Retrieval Method

The retrieval engine is built on the **BM25** algorithm, implemented via the lightweight and fast `bm25s` library. 

* **Mechanism**: BM25 relies on term frequency-inverse document frequency (TF-IDF) principles but adds saturation and document length normalization.
* **Ranking**: When a query is given, the engine removes English stopwords, tokenizes the remaining terms, and calculates a similarity score for every indexed chunk, returning the top `k` most statistically relevant segments.

---

## Performance Analysis

System performance is tracked via **Recall@k** (k=1, 3, 5, 10). 

* **Recall Calculation**: A source is considered "found" if the retrieved snippet character range overlaps by at least 5% with the correct ground truth source. 
* **Throughput**: The pipeline includes batch generation (`generate_batch`) which significantly optimizes GPU usage for dataset answering by processing 2 queries concurrently with `padding=True` and `truncation=True`. 

---

## Design Decisions

* **Pydantic Validation**: Ensures robust data serialization/deserialization, mitigating errors when interacting with complex nested JSON datasets.
* **CLI Construction**: Chosen `fire` over `argparse` for its capability to seamlessly wrap full Python classes into complex CLI commands with minimal boilerplate.
* **Memory Management**: The `Qwen3` model is loaded with `torch_dtype=torch.bfloat16` and `low_cpu_mem_usage=True` to prevent VRAM overflow during cold starts and batch processing.

---

## Challenges Faced

1. **Context Window Limitations**: Passing too many chunks directly to the LLM exceeded token limits. 
   * *Solution*: Restricted context insertion in `generate_message` to only the top 3 documents (`sources[:3]`), truncating them to 500 characters each to maintain safety margins.
2. **Batch Processing Errors**: Running individual sequential inferences was too slow for the 1000-question requirement.
   * *Solution*: Implemented batch tokenization in `generator.py` taking slices of size 2, handling padding properly, and recovering gracefully from batch-specific exceptions with empty answers.

---

## Instructions & Example Usage

The project is managed via the `uv` package manager and executed via a python module.

**1. Installation**
    
    make install

**2. Indexing the Knowledge Base**
    
    uv run python -m student index --max_chunk_size=2000

**3. Searching for a Query**
    
    uv run python -m student search "How to configure OpenAI server?" --k 5

**4. End-to-End Answer Generation**
    
    uv run python -m student answer "What method needs to be overridden in BaseProcessingInfo?" --k 5

**5. Evaluating Dataset Accuracy**
    
    uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json
    uv run python -m student evaluate --student_answer_path data/output/search_results/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json

---

## Resources

* **LangChain Documentation (Text Splitters)**: For managing chunking and document segmentation strategies.
* **BM25s Library**: For high-performance, lightweight BM25 indexing in Python.
* **HuggingFace Transformers**: Used for loading and managing the `Qwen` models.
* **AI Usage**: Generative AI tools were utilized to understand complex `transformers` and create the complete README.md file.