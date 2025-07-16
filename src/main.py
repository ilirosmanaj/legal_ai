import logging
from pathlib import Path

from src.agents.document_parser import DocumentParser


def setup_logging():
    """Configure logging with appropriate format and level"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def main():
    """Main function to process sample documents"""
    logger = setup_logging()

    sample_docs_dir = Path("docs/sample_documents/dismissal_protection_suits")

    if not sample_docs_dir.exists():
        logger.error(f"Sample documents directory '{sample_docs_dir}' not found.")
        return

    parser = DocumentParser()
    markdown_files = list(sample_docs_dir.rglob("*.md"))

    if not markdown_files:
        logger.warning(f"No markdown files found in {sample_docs_dir}")
        return

    logger.info(f"Found {len(markdown_files)} markdown files to process:")
    for file_path in markdown_files:
        logger.info(f"  - {file_path}")

    for file_path in markdown_files[:1]:
        try:
            logger.info(f"Processing: {file_path}")
            with open(file_path, encoding="utf-8") as f:
                markdown_text = f.read()

            parser.parse_document(document_name=file_path.name, markdown_text=markdown_text)
            logger.info(f"Successfully parsed: {file_path.name}")

        except FileNotFoundError:
            logger.error(f"Could not read file {file_path}")
        except PermissionError:
            logger.error(f"Permission denied reading file {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
