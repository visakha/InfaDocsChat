import os
import logging
from typing import List

from langchain_core.vectorstores import VectorStore

from my_utils.pdf2txt_converter import PDFTextExtractor
from my_utils.write_text_chunks_to_file import write_text_chunks_to_file, remove_before_tag, read_text_chunks_from_file, \
    store_text_chunks_in_vector_db, create_retrieval_qa

# Example usage (optional):
if __name__ == "__main__":
    extractor = PDFTextExtractor()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    path_to_docs_dir = 'C:/Users/vamsi/infa-docs/cdq/'
    path_to_llm_dir = 'C:/Users/vamsi/infa-docs/LLM-CDQ/'
    file_name_without_extn = 'IICS_October2024_GettingStarted_en'
    try:
        pdf_full_file_path = path_to_docs_dir + file_name_without_extn + '.pdf'
        extracted_text = extractor.extract_text(pdf_full_file_path)  # Replace "example.pdf" with your file
        # logger.info(extracted_text)
        extracted_text = remove_before_tag('C h a p t e r  1', extracted_text)
        text_chunks = [extracted_text]
        write_text_chunks_to_file(path_to_llm_dir+ file_name_without_extn + '.txt', text_chunks)
        chunks : List[str] = read_text_chunks_from_file(path_to_llm_dir + file_name_without_extn + '.txt')
        vecStore : VectorStore = store_text_chunks_in_vector_db(chunks, 100)
        qa_chain = create_retrieval_qa(vecStore)
        query = "How does the authentication process work?"
        response = qa_chain.run(query)
        print(response)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


