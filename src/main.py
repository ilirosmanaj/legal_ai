import json
from pathlib import Path
from typing import Dict

from src.agents.document_generator import DocumentGenerator
from src.agents.document_parser import DocumentParser
from src.logger import logger

CACHE_RESULTS = True
USE_CACHE = True
CACHE_DIR = "cache"

from enum import Enum


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value  # or str(obj)
        return super().default(obj)


def _save_parsed_document(file_path: Path, result: Dict, markdown_text: str):
    with open(f"{CACHE_DIR}/parsed_documents/{file_path.name}.json", "w") as f:
        json.dump({"parsed": result, "raw": markdown_text}, f, indent=2)


def _save_document_patterns(file_path: Path, result: Dict):
    with open(f"{CACHE_DIR}/document_patterns/{file_path.name}.json", "w") as f:
        json.dump(result, f, indent=2)


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

    if USE_CACHE:
        with open("cache/parsed_documents/dps_001.md.json") as f:
            data = json.load(f)
            parsed_documents.append(data["parsed"])
            raw_documents.append(data["raw"])
        with open("cache/parsed_documents/dps_004.md.json") as f:
            data = json.load(f)
            parsed_documents.append(data["parsed"])
            raw_documents.append(data["raw"])
        with open("cache/parsed_documents/dps_002.md.json") as f:
            data = json.load(f)
            parsed_documents.append(data["parsed"])
            raw_documents.append(data["raw"])
        with open("cache/parsed_documents/dps_003.md.json") as f:
            data = json.load(f)
            parsed_documents.append(data["parsed"])
            raw_documents.append(data["raw"])
        with open("cache/parsed_documents/dps_005.md.json") as f:
            data = json.load(f)
            parsed_documents.append(data["parsed"])
            raw_documents.append(data["raw"])

    else:
        for file_path in markdown_files:
            if "001" in file_path.name:
                continue

            if "004" in file_path.name:
                continue

            if "005" in file_path.name:
                continue

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
                    indent=2,
                )
                logger.info(f"Saved parsed document: {file_path.name}")
                logger.info("-" * 100)

    # pattern_detector = PatternDetector()

    # pattern_spec = pattern_detector.detect_patterns(parsed_documents, raw_documents)
    # logger.info(f"Pattern specification: {pattern_spec}")
    # with open("cache/document_patterns/pattern_spec.json", "w") as f:
    #     json.dump(pattern_spec, f, indent=2)
    # logger.info("Saved pattern specification")
    # logger.info("-" * 100)

    with open("cache/document_patterns/pattern_spec.json") as f:
        pattern_spec = json.load(f)

    with open("cache/templates/template.json") as f:
        template = json.load(f)

    # template_generator = TemplateBuilder()

    # import pdb; pdb.set_trace()
    # template = template_generator.build_template(pattern_spec, parsed_documents, raw_documents)
    # logger.info(f"Template: {template}")
    # with open("cache/templates/template.json", "w") as f:
    #     json.dump(template, f, cls=EnumEncoder, indent=2)

    document_generator = DocumentGenerator()
    variables = {}
    for variable in list(template["variable_system"]["variables"].values()):
        variables[variable["name"]] = input(f"Enter {variable['name']}, desc: {variable['description']}: ")
    document = document_generator.generate_document(template, variables)
    logger.info(f"Document: {document}")

    with open("cache/generated_documents/generated_document_001.md", "w") as f:
        f.write(document)


if __name__ == "__main__":
    main()
