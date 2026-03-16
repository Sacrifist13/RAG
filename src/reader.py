import sys
from pathlib import Path
from tqdm import tqdm
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .models import MinimalSource


class Reader:
    """
    Reader class for splitting files in a repo into minimal sources.

    Methods
    -------
    read_split(repo_path: Path, max_chunk_size: int) ->
    List[MinimalSource] | None
        Splits files in repo_path into chunks of max_chunk_size.

        Args:
            repo_path (Path): Path to the repository.
            max_chunk_size (int): Max size of each chunk.

        Returns:
            List[MinimalSource] | None: List of chunked sources or None if
            empty.
    """

    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def read_split(
        self, repo_path: Path, max_chunk_size: int
    ) -> List[MinimalSource] | None:
        """
        Splits files in a repo into chunks for processing.

        Args:
            repo_path (Path): Path to the repository directory.
            max_chunk_size (int): Maximum size of each chunk.

        Returns:
            List[MinimalSource] | None: List of chunked sources or None if
            empty.
        """

        overlap = int(max_chunk_size * (10 / 100))

        py_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON,
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )

        md_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )

        co_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                ".",
                " ",
                "",
            ],
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )

        files = [file for file in repo_path.rglob("*") if file.is_file()]

        if not files:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Datas directory is empty: "
                f"{repo_path}{self.RESET}\n",
                file=sys.stderr,
            )
            return None

        datas_formated: List[MinimalSource] = []

        for file in tqdm(files, desc="Processing repository"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()

                if file.suffix == ".py":
                    docs = py_splitter.create_documents([content])
                elif file.suffix == ".md":
                    docs = md_splitter.create_documents([content])
                else:
                    docs = co_splitter.create_documents([content])

                for doc in docs:
                    sub_content = doc.page_content
                    start_index = doc.metadata["start_index"]
                    end_index = start_index + len(sub_content)

                    datas_formated.append(
                        MinimalSource(
                            file_path=str(file),
                            first_character_index=start_index,
                            last_character_index=end_index,
                            content=sub_content,
                        )
                    )
            except Exception:
                pass

        if not datas_formated:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Datas directory is empty: "
                f"{repo_path}{self.RESET}\n",
                file=sys.stderr,
            )
            return None

        return datas_formated
