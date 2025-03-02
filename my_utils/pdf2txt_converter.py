from typing import List
import fitz  # PyMuPDF


class PDFTextExtractor:
    """
    A class for extracting text from PDF files using PyMuPDF (fitz).
    """

    def __init__(self) -> None:
        """
        Initializes the PDFTextExtractor.
        """
        pass  # No specific initialization needed for now.

    def extract_text(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file.

        Args:
            pdf_path: The path to the PDF file.

        Returns:
            The extracted text as a string.

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
            Exception: if there is an issue with fitz
        """
        try:
            with fitz.open(pdf_path) as doc:
                text_pages: List[str] = []
                for page in doc:
                    text_pages.append(page.get_text())
                return "".join(text_pages)

        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
        except Exception as e:
            raise Exception(f"Error processing PDF with fitz: {e}")



    def extract_text_from_multiple_pdfs(self, pdf_paths: List[str]) -> List[str]:
        """
        Extracts text from multiple PDF files.

        Args:
            pdf_paths: A list of paths to the PDF files.

        Returns:
            A list of extracted texts, each corresponding to a PDF file.

        Raises:
            FileNotFoundError: If any of the specified PDF files does not exist.
            Exception: if there is an issue with fitz
        """
        extracted_texts: List[str] = []
        for pdf_path in pdf_paths:
            extracted_texts.append(self.extract_text(pdf_path))
        return extracted_texts


