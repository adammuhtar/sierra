# Standard library imports
import io
import logging
from pathlib import Path
import re
from typing import Dict, List
from urllib.parse import urlparse

# Third party imports
from PIL import Image
import pymupdf
import requests
import torch
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_files(urls: List[Dict[str, list]], folder_path: Path) -> None:
    """
    Downloads a list of files from the specified URLs into a given folder.

    Args:
        * urls (`List[str]`): URLs of the files to download.
        * folder_path (`Path`): The Path object for the folder where the files
            will be saved.

    Creates a folder if it doesn't exist and downloads each file from the list into it.
    """
    # Create the folder if it doesn't exist
    folder_path.mkdir(parents=True, exist_ok=True)

    # Loop through the list of files and download each one
    for company in urls:
        for url in tqdm(
            iterable=list(company.values())[0],
            desc=f"Downloading files from {str(list(company.keys())[0])}",
            unit="file",
            total=len(list(company.values())[0])
        ):
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raises HTTPError if the HTTP request returned an unsuccessful status code
                
                # Create a folder for company if it doesn't exist
                company_name = str(list(company.keys())[0])
                company_folder_path = folder_path / company_name
                company_folder_path.mkdir(parents=True, exist_ok=True)

                # Extract the file name from the URL
                file_name = Path(urlparse(url).path).name
                if not file_name.endswith(".pdf"):
                    file_name += ".pdf"

                # Write the file to the specified folder
                if (company_folder_path / file_name).exists():
                    logging.info(f"{file_name} already exists in {company_name} folder.")
                    continue
                else:
                    with open(company_folder_path / file_name, "wb") as file:
                        file.write(response.content)

            except requests.HTTPError as e:
                print(f"HTTP Error occurred: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")


def pdf_to_image(doc: pymupdf.Document, page_num: int) -> Image.Image:
    """
    Converts a PDF page to an image.
    
    Args:
        * doc (`pymupdf.Document`): The PDF document to convert.
        * page_num (`int`): The page number to convert.
    
    Returns:
        `Image.Image`: The converted image in PIL Image format.
    """
    # Check if the page number is within the valid range
    if page_num < 0 or page_num >= len(doc):
        raise ValueError("Page number is out of range")

    # Get the specified page
    page = doc.load_page(page_num)

    # Render the page to an image (pixmap)
    pix = page.get_pixmap()

    # Convert the pixmap to an image (Pillow Image)
    img = Image.frombytes(
        mode="RGB", size=[pix.width, pix.height], data=pix.samples
    )

    return img


def preprocess_text(
        text: str,
        encoding: bool = True,
        lowercase: bool = False,
        remove_newlines: bool = True
    ) -> str:
        """
        Preprocess the input text by cleaning and normalizing it.

        Args:
            * text (`str`): Text to preprocess.
            * encoding (`bool`, optional): Convert non-UTF-8 characters to UTF-8.
                Defaults to True.
            * lowercase (`bool`, optional): Convert entire text to lowercase.
                Defaults to False.
            * remove_newlines (`bool`, optional): Remove newline characters.
                Defaults to True.

        Returns:
            `str`: Preprocessed text.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        
        # Fix apostrophes/quotation marks
        _text = re.sub("[‘’]", "'", text)
        _text = re.sub("[“”]", '"', _text)

        if encoding:
            _text = re.sub("(&\\\\#x27;|&#x27;)", "'", _text)
        
        # Remove newlines, tabs, non-breaking spaces, excess backslashes/whitespaces
        if remove_newlines:
            _text = re.sub("[\n\r]+", " ", _text)
        _text = re.sub("[\t\xa0]+", " ", _text)
        _text = re.sub(r"\\+", "", _text)
        _text = re.sub(r"\s+", " ", _text).strip()

        if lowercase:
            _text = _text.lower()

        return _text

# def extract_table(page: pymupdf.Page):
    