"""
Interactive FAISS‑index builder
-------------------------------

* Prompts the user to choose the language set: 'ja' or 'en'.
* Loads .txt / .md / .pdf files from rag_docs/<lang>.
* Saves the index to index/<lang>.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

# --------------------------------------------------------------------------- #
# Low‑level I/O helpers
# --------------------------------------------------------------------------- #


def iter_paths(pattern: str | Path) -> Iterable[Path]:
    """Yield Path objects that match a glob pattern."""
    pattern = str(pattern)
    if ("*" not in pattern) and ("?" not in pattern):
        p = Path(pattern)
        return [p] if p.exists() else []
    root, glob_pat = os.path.split(pattern)
    root_path = Path(root or ".").resolve()
    return root_path.rglob(glob_pat)


def load_text_files(pattern: str | Path) -> List[Document]:
    """Load .txt /.md files into LangChain Documents."""
    docs: List[Document] = []
    for path in iter_paths(pattern):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            content = f.read()
        docs.append(
            Document(
                page_content=content,
                metadata={"source": str(path), "title": path.name, "page_no": 1},
            )
        )
    return docs


def load_pdf_files(pattern: str | Path) -> List[Document]:
    """Load PDFs, turning each page into its own Document."""
    docs: List[Document] = []
    page_no = 1
    for path in iter_paths(pattern):
        if not path.is_file():
            continue
        reader = PdfReader(path)
        for page in reader.pages:
            docs.append(
                Document(
                    page.extract_text() or "",
                    metadata={
                        "source": str(path),
                        "title": path.name,
                        "page_no": page_no,
                    },
                )
            )
            page_no += 1
    return docs


# --------------------------------------------------------------------------- #
# Index‑building logic
# --------------------------------------------------------------------------- #


def build_index(
    input_dir: Path,
    out_dir: Path,
    embed_model_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> None:
    """Read files, split into chunks, embed, and save a FAISS index."""
    txt_docs = load_text_files(input_dir / "*.txt")
    md_docs = load_text_files(input_dir / "*.md")
    pdf_docs = load_pdf_files(input_dir / "*.pdf")

    total_docs = txt_docs + md_docs + pdf_docs
    if not total_docs:
        print("⚠️  No matching files found; aborting.")
        return

    print(
        "Loaded:",
        f"txt={len(txt_docs)}, md={len(md_docs)}, pdf={len(pdf_docs)}, total={len(total_docs)}",
    )

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        disallowed_special=(),
    )
    chunks = splitter.transform_documents(total_docs)

    print("Building FAISS index ...")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    index = FAISS.from_documents(chunks, embeddings)

    out_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(out_dir)
    print(f"✅ Index saved to {out_dir}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    load_dotenv()
    HUG_EMBE_MODEL_NAME = os.getenv("HUG_EMBE_MODEL_NAME")
    if not HUG_EMBE_MODEL_NAME:
        raise EnvironmentError("HUG_EMBE_MODEL_NAME is missing in .env")

    # Ask the user which language corpus to process
    lang = input("Choose language to index ('ja' or 'en'): ").strip().lower()
    if lang not in {"ja", "en"}:
        raise ValueError("Invalid choice. Please enter 'ja' or 'en'.")

    INPUT_DIR = Path("rag_docs") / lang
    OUT_DIR = Path("src/routers/agentic_rag/index") / lang

    print(f"🚀 Building index for '{lang}' corpus …")
    build_index(INPUT_DIR.resolve(), OUT_DIR.resolve(), HUG_EMBE_MODEL_NAME)
