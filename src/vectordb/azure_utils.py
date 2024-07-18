# Standard library imports
import base64
import configparser
import logging
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

# Third party imports
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient


# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_client(config_file: str = "client.ini") -> DocumentIntelligenceClient:
    """
    Creates and returns an instance of DocumentIntelligenceClient using credentials from a configuration file.

    Reads the API key and endpoint from the specified configuration file under the 'DocumentAI' section.

    Args:
        config_file (str): The path to the configuration file. Default is 'client.ini'.

    Returns:
        DocumentIntelligenceClient: A client for interacting with the Azure Document Intelligence API.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If the required keys ('api_key', 'endpoint') are not found in the configuration file.
        configparser.Error: If there is an error in parsing the configuration file.
    """
    config = configparser.ConfigParser()
    config_path = Path(config_file)

    # Check if the configuration file exists
    if not config_path.is_file():
        logger.error("The configuration file '%s' was not found.", config_file)
        raise FileNotFoundError(
            f"The configuration file '{config_file}' was not found."
        )

    try:
        # Read configuration file
        config.read(config_path)

        # Check if the 'DocumentAI' section exists
        if "DocumentAI" not in config:
            logger.error(
                "The 'DocumentAI' section is missing in the configuration file."
            )
            raise KeyError(
                "The 'DocumentAI' section is missing in the configuration file."
            )

        # Get API key and endpoint from the configuration file
        api_key = config.get("DocumentAI", "api_key", fallback=None)
        endpoint = config.get("DocumentAI", "endpoint", fallback=None)

        if not api_key or not endpoint:
            logger.error(
                "The required 'api_key' or 'endpoint' is missing in the configuration file."
            )
            raise KeyError(
                "The required 'api_key' or 'endpoint' is missing in the configuration file."
            )

        # Create and return the DocumentIntelligenceClient instance
        client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )
        logger.info("Successfully created DocumentIntelligenceClient.")
        return client
    except configparser.Error as e:
        logger.error("Error parsing the configuration file '%s': %s", config_file, e)
        raise configparser.Error(
            f"Error parsing the configuration file '{config_file}': {e}"
        )


def load_file_as_base64(file_path: str) -> str:
    """
    Loads a file and encodes its content as a base64 string.

    Args:
        file_path (`str`): The path to the file to be encoded.

    Returns:
        `str`: The base64 encoded string of the file content.

    Raises:
        `FileNotFoundError`: If the specified file does not exist.
        `PermissionError`: If there are permissions issues while reading the file.
        `IOError`: If an I/O error occurs while reading the file.
    """
    path = Path(file_path)

    # Check if the file exists
    if not path.is_file():
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    try:
        # Read the file and encode its content as base64
        with path.open("rb") as f:
            data = f.read()
        base64_bytes = base64.b64encode(data)
        base64_string = base64_bytes.decode("utf-8")
        return base64_string
    except PermissionError:
        raise PermissionError(
            f"Permission denied while accessing the file '{file_path}'."
        )
    except IOError as e:
        raise IOError(f"An error occurred while reading the file '{file_path}': {e}")