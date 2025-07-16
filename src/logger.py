import logging


def setup_logging():
    """Configure logging with appropriate format and level"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("legal_document_parser")


logger = setup_logging()
