import sys
from pathlib import Path
from tqdm import tqdm
from typing import List
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class MinimalSource(BaseModel):
    file_path: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)
    content: str = Field(min_length=1)


class Reader:

    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def read_split(
        self, repo_path: Path, max_chunk_size: int
    ) -> List[MinimalSource] | None:

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

        files = [
            file
            for file in repo_path.rglob("*")
            if file.is_file()
            if "__pycache__" not in file.parts
            if ".git" not in file.parts
        ]

        if not files:
            print(
                f"\n{self.RED}{self.BOLD}❌ [ERROR] Datas directory is empty: "
                f"{repo_path.name}{self.RESET}\n",
                file=sys.stderr,
            )

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
                            file_path=str(file.relative_to(repo_path)),
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
                f"{repo_path.name}{self.RESET}\n",
                file=sys.stderr,
            )
            return None

        return datas_formated
