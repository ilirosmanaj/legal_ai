import json
from pathlib import Path
from typing import Dict

from src.agents.document_parser import DocumentParser
from src.agents.pattern_detector import PatternDetector
from src.logger import logger

CACHE_RESULTS = True
CACHE_DIR = "cache"


def _save_parsed_document(file_path: Path, result: Dict, markdown_text: str):
    with open(f"{CACHE_DIR}/{file_path.name}.json", "w") as f:
        json.dump(
            {"parsed": result, "raw": markdown_text},
            f,
        )


def parse_document(file_path: Path, parser: DocumentParser):
    try:
        logger.info(f"Processing: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            markdown_text = f.read()

        result = parser.parse_document(document_name=file_path.name, markdown_text=markdown_text)
        logger.info(f"Successfully parsed: {file_path.name}")
        if CACHE_RESULTS:
            _save_parsed_document(file_path, result, markdown_text)

        return result, markdown_text
    except FileNotFoundError:
        logger.error(f"Could not read file {file_path}")
    except PermissionError:
        logger.error(f"Permission denied reading file {file_path}")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)


def main():
    """Main function to process sample documents"""

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

    raw_documents = []
    parsed_documents = []

    if False:
        with open("dps_001.md.json") as f:
            data = json.load(f)
            parsed_documents.append(data["parsed"])
            raw_documents.append(data["raw"])
        with open("dps_004.md.json") as f:
            data = json.load(f)
            parsed_documents.append(data["parsed"])
            raw_documents.append(data["raw"])
    else:
        for file_path in markdown_files:
            result, markdown_text = parse_document(file_path, parser)

            if not result:
                logger.error(f"Failed to parse document: {file_path}")
                continue

            raw_documents.append(markdown_text)
            parsed_documents.append(result)

            with open(f"{file_path.name}.json", "w") as f:
                json.dump(
                    {
                        "parsed": result,
                        "raw": markdown_text,
                    },
                    f,
                )
                logger.info(f"Saved parsed document: {file_path.name}")
                logger.info("-" * 100)

    pattern_detector = PatternDetector()

    pattern_spec = pattern_detector.detect_patterns(parsed_documents, raw_documents)
    logger.info(f"Pattern specification: {pattern_spec}")


if __name__ == "__main__":
    main()
