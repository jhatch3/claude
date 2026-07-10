import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Windows consoles default to cp1252, which can't encode chars like the
# zero-width space (​) found in many PDFs. Force UTF-8 output.
sys.stdout.reconfigure(encoding="utf-8")

# Directory this script lives in, so paths work from any cwd.
HERE = os.path.dirname(os.path.abspath(__file__))


def load_pdf(file_path: str) -> list:
    """
    Load a PDF file and return its content as a list of documents.
    
    input example:
        file_path = "path/to/your/file.pdf"

    output example:
        [Document(page_content="..."), Document(page_content="...")]

    """
    loader = PyMuPDFLoader(file_path=file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def split_documents(documents, breakpoint_threshold_type="percentile") -> list:
    """
    Split documents into semantically coherent chunks.

    Embeds each sentence and cuts where sentence-to-sentence meaning shifts,
    rather than at a fixed character count. `breakpoint_threshold_type` controls
    where cuts happen: "percentile" (default), "standard_deviation",
    "interquartile", or "gradient".
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
    )
    split_docs = text_splitter.split_documents(documents)

    print(f"Split into {len(split_docs)} chunks")
    return split_docs


def main():
    """
    Main function to load a PDF file and print its content.
    """
    file_path = os.path.join(HERE, "docs", "Justin-Hatch-S26-Resume (2).pdf")
    documents = load_pdf(file_path)
    split_docs = split_documents(documents)

    for i, doc in enumerate(split_docs):
        print(f"Chunk {i + 1}:\n{doc.page_content}\n{'-' * 40}")


if __name__ == "__main__":
    main()
