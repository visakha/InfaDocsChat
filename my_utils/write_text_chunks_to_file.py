import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStore
# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def write_text_chunks_to_file(file_path: str, text_chunks: List[str]) -> None:
    """Writes the provided text chunks to a specified file.
    Args:
        file_path (str): The path to the file where the text chunks should be written.
        text_chunks (List[str]): A list of text chunks to be written to the file.
    """
    if not text_chunks:
        logging.warning("No text chunks provided. Output file will be empty.")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(text_chunks) + "\n\n")
    except OSError as e:
        logging.error(f"File operation error for {file_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in write_text_chunks_to_file: {e}")


def read_text_chunks_from_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Reads a text file and splits its content into chunks using RecursiveCharacterTextSplitter.
    Args:
        file_path (str): The path to the text file to be read.
        chunk_size (int, optional): The maximum size of each text chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap size between consecutive chunks. Defaults to 200.
    Returns:
        List[str]: A list of text chunks.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If an error occurs while reading the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_text(text)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {file_path}")
    except Exception as e:
        logging.error(f"Error while reading and splitting text from file: {e}")
        raise


def remove_before_tag(tag_name: str, input_string: str) -> str:
    """
    Removes everything before the specified tag in the input string.
    Args:
        tag_name (str): The tag to search for.
        input_string (str): The input string to process.
    Returns:
        str: A string with all content before the tag (inclusive) removed.
            If the tag is not found, the original string is returned.
    """
    try:
        tag_index = input_string.find(tag_name)
        if tag_index != -1:
            return input_string[tag_index + len(tag_name):]
        else:
            logging.warning(f"Tag '{tag_name}' not found in input string.")
            return input_string
    except Exception as e:
        logging.error(f"Error in remove_before_tag: {e}")
        return input_string


def store_text_chunks_in_vector_db(text_chunks: List[str], vector_dimension: int) -> VectorStore:
    """
    Stores text chunks in a FAISS vector database.
    Args:
        text_chunks (List[str]): A list of text chunks to be encoded and stored.
        vector_dimension (int): Dimension of the vectors to be stored in FAISS.
    Returns:
        VectorStore: The FAISS vector store object.
    Raises:
        ValueError: If text_chunks is empty.
        Exception: If an error occurs during vectorization or FAISS index creation.
    """
    if not text_chunks:
        raise ValueError("No text chunks provided to store in the vector database.")
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=OpenAIEmbeddings())
        vector_store.save_local("vector_db")
        return vector_store
    except Exception as e:
        logging.error(f"Error storing text chunks in vector database: {e}")
        raise


def create_retrieval_qa(vector_store: VectorStore) -> RetrievalQA:
    """
    Creates a RetrievalQA chain from a given vector store.
    Args:
        vector_store (VectorStore): The vector store to use for retrieval.
    Returns:
        RetrievalQA: A RetrievalQA chain configured to use the provided vector store.
    Raises:
        Exception: If an error occurs while creating the RetrievalQA chain.
    """
    try:
        llm = OpenAI(model_name="gpt-4")
        retriever = vector_store.as_retriever()
        qa_chain: RetrievalQA = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever
        )
        return qa_chain
    except Exception as e:
        logging.error(f"Error creating RetrievalQA: {e}")
        raise