import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.clients import LLMClient

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    DATE = "date"
    MONEY = "money"
    LEGAL_REF = "legal_reference"
    CASE_NUMBER = "case_number"
    COURT = "court"
    ADDRESS = "address"
    POSITION = "position"
    OTHER = "other"


@dataclass
class Entity:
    text: str
    entity_type: str  # EntityType
    start: int
    end: int
    confidence: float
    role: Optional[str] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    normalized: Optional[str] = None


@dataclass
class Section:
    id: str
    title: str
    level: int
    content: str
    purpose: Optional[str] = None
    subsections: List["Section"] = None
    entities: List[Entity] = None

    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []
        if self.entities is None:
            self.entities = []


class DocumentParser:
    """
    Advanced Document Parser using LLMs for intelligent analysis.
    Handles markdown legal documents with deep understanding.
    """

    def __init__(self):
        """Initialize the LLM Document Parser"""
        self.llm = self._initialize_llm()
        self.log_raw_response = True

    def _initialize_llm(self, temperature: float = 0):
        """Initialize LLM client using the shared LLMClient class"""
        return LLMClient(temperature=temperature)

    def parse_document(self, document_name: str, markdown_text: str) -> Dict[str, Any]:
        """
        Parse a legal document using LLM for deep understanding.

        Args:
            markdown_text: The markdown content to parse

        Returns:
            Comprehensive structured representation
        """

        logger.info(f"Starting parsing for document {document_name}")

        try:
            doc_analysis = self._analyze_document(markdown_text)
            logger.info("Document analysis done!")

            structure = self._extract_structure(markdown_text, doc_analysis)
            logger.info("Structure extraction done!")

            entities = self._extract_entities(structure)
            logger.info("Entities extraction done!")

            metadata = self._extract_metadata(markdown_text, doc_analysis)
            logger.info("Metadata extraction done!")

            relationships = self._analyze_relationships(entities)
            logger.info("Relationships analysis done!")

            insights = self._generate_insights(structure, entities, relationships)
            logger.info("Insights generation done!")

            output = {
                "doc_id": document_name,
                "type": doc_analysis["document_type"],
                "metadata": metadata,
                "structure": self._structure_to_dict(structure),
                "entities": self._organize_entities(entities),
                "relationships": relationships,
                "insights": insights,
                "parser_metadata": {
                    "parsing_time": datetime.now().isoformat(),
                    "parser_version": "2.0-LLM",
                    "llm_model": os.getenv("CLAUDE_MODEL", "claude-opus-4-20250514"),
                    "confidence_scores": doc_analysis["confidence_scores"],
                    "extraction_method": "llm_deep_parsing",
                },
            }

            logger.info(f"Successfully parsed document {document_name}")
            return output

        except Exception as e:
            logger.error(f"Error parsing document {document_name}: {e!s}")
            raise

    def _analyze_document(self, markdown_text: str) -> Dict[str, Any]:
        """Initial document analysis to understand type and structure"""

        prompt = f"""
        Analyze this legal document and provide a comprehensive initial assessment.

        Document:
        {markdown_text}

        Provide analysis in JSON format:
        {{
            "document_type": "specific type (e.g., dismissal_protection_suit)",
            "jurisdiction": "country/state",
            "language": "primary language",
            "parties": {{
                "plaintiff": "name if identifiable",
                "defendant": "name if identifiable"
            }},
            "key_dates": ["list of important dates mentioned"],
            "legal_domain": "area of law (employment, contract, etc.)",
            "document_purpose": "what this document aims to achieve",
            "structural_style": "formal/semi-formal/informal",
            "completeness": "complete/partial/draft",
            "confidence_scores": {{
                "type_identification": 0.95,
                "structure_clarity": 0.90,
                "content_quality": 0.88
            }}
        }}
        """
        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_analyze_document]: {response}")

        return json.loads(response)

    def _extract_structure(self, markdown_text: str, doc_analysis: Dict) -> Section:
        """Extract document structure with semantic understanding"""

        prompt = f"""
        Extract the hierarchical structure of this legal document with semantic understanding.

        Document type: {doc_analysis['document_type']}

        Document:
        {markdown_text}

        For each section, provide:
        1. The hierarchical structure (levels based on # headers)
        2. The semantic purpose of each section
        3. Key information contained
        4. How sections relate to each other

        Return as JSON:
        {{
            "root": {{
                "title": "Document Root",
                "purpose": "overall document purpose",
                "sections": [
                    {{
                        "id": "unique_id",
                        "title": "section title",
                        "level": 1,
                        "purpose": "what this section accomplishes",
                        "key_content": ["main points"],
                        "subsections": [...]
                    }}
                ]
            }}
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_extract_structure]: {response}")
        structure_data = json.loads(response)

        return self._build_section_tree(structure_data["root"], markdown_text)

    def _extract_entities(self, structure: Section) -> List[Entity]:
        """Extract entities with deep contextual understanding"""

        entities = []
        for section in self._flatten_sections(structure):
            if not section.content:
                continue

            prompt = f"""
            Extract all legal entities from this section with their roles and relationships.

            Section: {section.title}
            Content: {section.content}

            For each entity, identify:
            1. Entity text (exact as appears)
            2. Entity type (person, organization, date, money, legal_reference, case_number, court, address, position)
            3. Role in the document (plaintiff, defendant, attorney, judge, employer, etc.)
            4. Context (what the entity refers to - e.g., "hire_date", "dismissal_date", "monthly_salary")
            5. Relationships to other entities
            6. Confidence score (0-1)

            Also normalize:
            - Dates to YYYY-MM-DD format
            - Money to numerical amount with currency
            - Legal references with full law name

            Return as JSON:
            {{
                "entities": [
                    {{
                        "text": "Maria Schmidt",
                        "type": "person",
                        "role": "plaintiff",
                        "context": "employee filing suit",
                        "normalized": null,
                        "relationships": ["employed_by: TechCorp GmbH"],
                        "confidence": 0.95,
                        "position_hint": "found in PLAINTIFF section"
                    }}
                ]
            }}
            """

            response = self.llm.complete(prompt)
            if self.log_raw_response:
                logger.info(f"Raw response [_extract_entities]: {response}")
            section_entities = json.loads(response)

            for ent_data in section_entities["entities"]:
                start = section.content.find(ent_data["text"])
                if start == -1:
                    start = 0
                end = start + len(ent_data["text"])

                entity = Entity(
                    text=ent_data["text"],
                    entity_type=ent_data["type"].lower(),
                    start=start,
                    end=end,
                    confidence=ent_data["confidence"],
                    role=ent_data.get("role"),
                    context=ent_data.get("context"),
                    normalized=ent_data.get("normalized"),
                    metadata={"relationships": ent_data.get("relationships", []), "section": section.title},
                )
                entities.append(entity)

        return self._resolve_entity_conflicts(entities)

    def _extract_metadata(self, markdown_text: str, doc_analysis: Dict) -> Dict[str, Any]:
        """Extract comprehensive metadata using LLM understanding"""

        prompt = f"""
        Extract all metadata from this legal document.

        Document type: {doc_analysis['document_type']}

        Document header (first 1000 chars):
        {markdown_text[:1000]}

        Extract:
        1. Case information (number, court, filing date)
        2. Jurisdiction and applicable law
        3. Document status (draft, filed, final)
        4. Important deadlines mentioned
        5. Procedural information
        6. Any reference numbers or IDs

        Return as JSON:
        {{
            "case_number": "extracted case number",
            "court": {{
                "name": "court name",
                "location": "city/jurisdiction",
                "type": "labor/civil/criminal"
            }},
            "filing_date": "YYYY-MM-DD",
            "jurisdiction": "applicable jurisdiction",
            "applicable_laws": ["list of referenced laws"],
            "document_status": "draft/filed/final",
            "deadlines": [
                {{
                    "date": "YYYY-MM-DD",
                    "description": "what deadline"
                }}
            ],
            "reference_numbers": ["any other IDs"],
            "language": "primary language",
            "procedural_stage": "initial filing/appeal/etc"
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_extract_metadata]: {response}")
        metadata = json.loads(response)

        metadata["extraction_timestamp"] = datetime.now().isoformat()
        metadata["document_statistics"] = {
            "word_count": len(markdown_text.split()),
            "section_count": self._count_sections(doc_analysis.get("structure", {})),
        }

        return metadata

    def _analyze_relationships(self, entities: List[Entity]) -> Dict[str, Any]:
        """Analyze relationships between entities and document elements"""

        # Prepare entity data for analysis
        entity_data = [
            {
                "text": e.text,
                "type": e.entity_type,
                "role": e.role,
                "context": e.context,
                "section": e.metadata.get("section") if e.metadata else None,
            }
            for e in entities
        ]

        prompt = f"""
        Analyze relationships between entities in this legal document.

        Entities found:
        {json.dumps(entity_data, indent=2)}

        Identify:
        1. Employment relationships (who worked for whom)
        2. Legal representation (who represents whom)
        3. Temporal relationships (sequence of events)
        4. Causal relationships (what led to what)
        5. Financial relationships (who owes/paid whom)
        6. Procedural relationships (who filed against whom)

        Return as JSON:
        {{
            "employment_relationships": [
                {{
                    "employee": "person name",
                    "employer": "company name",
                    "position": "job title",
                    "duration": {{
                        "start": "YYYY-MM-DD",
                        "end": "YYYY-MM-DD"
                    }}
                }}
            ],
            "legal_relationships": [...],
            "temporal_sequence": [
                {{
                    "event": "description",
                    "date": "YYYY-MM-DD",
                    "entities_involved": ["list"]
                }}
            ],
            "causal_chain": [...],
            "financial_relationships": [...],
            "procedural_relationships": [...]
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_analyze_relationships]: {response}")

        return json.loads(response)

    def _generate_insights(self, structure: Section, entities: List[Entity], relationships: Dict) -> Dict[str, Any]:
        """Generate insights and potential issues in the document"""

        prompt = f"""
        Based on the document analysis, provide insights and identify potential issues.

        Document summary:
        - Type: {structure.purpose if hasattr(structure, 'purpose') else 'Legal document'}
        - Number of sections: {self._count_sections(structure)}
        - Key entities: {len(entities)} found
        - Relationships identified: {len(relationships.get('employment_relationships', []))} employment,
          {len(relationships.get('legal_relationships', []))} legal

        Provide insights on:
        1. Document completeness (missing sections or information)
        2. Legal consistency (any contradictions or issues)
        3. Risk factors (potential legal weaknesses)
        4. Strengths (well-argued points)
        5. Recommendations (what could be improved)
        6. Unusual patterns (anything atypical for this document type)

        Return as JSON:
        {{
            "completeness_assessment": {{
                "score": 0.9,
                "missing_elements": ["list of missing items"],
                "well_covered_areas": ["list of strong sections"]
            }},
            "consistency_check": {{
                "issues_found": ["list of inconsistencies"],
                "confidence": 0.85
            }},
            "risk_assessment": {{
                "high_risk_areas": ["list with explanations"],
                "medium_risk_areas": ["list"],
                "low_risk_areas": ["list"]
            }},
            "strengths": ["list of strong points"],
            "recommendations": [
                {{
                    "area": "section or aspect",
                    "suggestion": "specific improvement",
                    "priority": "high/medium/low"
                }}
            ],
            "unusual_patterns": ["anything atypical noticed"]
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_generate_insights]: {response}")

        return json.loads(response)

    def _resolve_entity_conflicts(self, entities: List[Entity]) -> List[Entity]:
        """Resolve conflicts between entities"""

        resolved_entities = entities.copy()
        return self._deduplicate_entities(resolved_entities)

    def _build_section_tree(self, section_data: Dict, full_text: str) -> Section:
        """Build Section tree from LLM response"""

        section = Section(
            id=section_data.get("id", self._generate_section_id(section_data["title"])),
            title=section_data["title"],
            level=section_data.get("level", 0),
            content=self._extract_section_content(section_data["title"], full_text),
            purpose=section_data.get("purpose"),
        )

        for subsection_data in section_data.get("sections", []):
            subsection = self._build_section_tree(subsection_data, full_text)
            section.subsections.append(subsection)

        return section

    def _extract_section_content(self, title: str, full_text: str) -> str:
        """Extract content for a specific section from full text"""

        lines = full_text.split("\n")
        content_lines = []
        in_section = False
        section_level = 0

        for line in lines:
            if title in line and line.strip().startswith("#"):
                in_section = True
                section_level = len(line.split()[0])  # Count #s
                continue

            if in_section and line.strip().startswith("#"):
                current_level = len(line.split()[0])
                if current_level <= section_level:
                    break

            if in_section:
                content_lines.append(line)

        return "\n".join(content_lines).strip()

    def _flatten_sections(self, section: Section) -> List[Section]:
        """Flatten section tree for processing"""

        sections = [section] if section.content else []
        for subsection in section.subsections:
            sections.extend(self._flatten_sections(subsection))
        return sections

    def _find_entity_conflicts(self, entities: List[Entity]) -> List[List[Entity]]:
        """Find potentially conflicting entities"""

        conflicts = []
        processed = set()

        for i, e1 in enumerate(entities):
            if i in processed:
                continue

            conflict_group = [e1]

            for j, e2 in enumerate(entities[i + 1 :], i + 1):
                if self._entities_overlap(e1, e2) or self._entities_similar(e1, e2):
                    conflict_group.append(e2)
                    processed.add(j)

            if len(conflict_group) > 1:
                conflicts.append(conflict_group)
                processed.add(i)

        return conflicts

    def _entities_overlap(self, e1: Entity, e2: Entity) -> bool:
        """Check if two entities overlap in position"""
        return not (e1.end <= e2.start or e2.end <= e1.start)

    def _entities_similar(self, e1: Entity, e2: Entity) -> bool:
        """Check if entities are similar enough to be conflicts"""

        if e1.text.lower() in e2.text.lower() or e2.text.lower() in e1.text.lower():
            return True

        if e1.normalized and e2.normalized and e1.normalized == e2.normalized:
            return True

        return False

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities intelligently"""

        # sort by confidence then by length (prefer more complete extractions)
        sorted_entities = sorted(entities, key=lambda e: (-e.confidence, -len(e.text)))

        deduped = []
        seen = set()

        for entity in sorted_entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                deduped.append(entity)
                seen.add(key)

        return deduped

    def _generate_section_id(self, title: str) -> str:
        """Generate section ID from title"""
        clean_title = re.sub(r"[^a-zA-Z0-9\s]", "", title)
        return clean_title.lower().replace(" ", "_")

    def _structure_to_dict(self, section: Section) -> Dict[str, Any]:
        """Convert Section tree to dictionary"""

        result = {
            "id": section.id,
            "title": section.title,
            "level": section.level,
            "purpose": section.purpose,
            "content": section.content[:200] + "..." if len(section.content) > 200 else section.content,
        }

        if section.subsections:
            result["subsections"] = [self._structure_to_dict(subsec) for subsec in section.subsections]

        return result

    def _organize_entities(self, entities: List[Entity]) -> Dict[str, List[Dict]]:
        """Organize entities by type"""

        organized = {
            "persons": [],
            "organizations": [],
            "dates": [],
            "monetary": [],
            "legal_refs": [],
            "case_numbers": [],
            "courts": [],
            "addresses": [],
            "positions": [],
        }

        type_mapping = {
            EntityType.PERSON: "persons",
            EntityType.ORGANIZATION: "organizations",
            EntityType.DATE: "dates",
            EntityType.MONEY: "monetary",
            EntityType.LEGAL_REF: "legal_refs",
            EntityType.CASE_NUMBER: "case_numbers",
            EntityType.COURT: "courts",
            EntityType.ADDRESS: "addresses",
            EntityType.POSITION: "positions",
        }

        type_mapping_2 = {
            "person": "persons",
            "organization": "organizations",
            "date": "dates",
            "money": "monetary",
            "legal_reference": "legal_refs",
            "case_number": "case_numbers",
            "court": "courts",
            "address": "addresses",
            "position": "positions",
        }

        for entity in entities:
            entity_dict = self._entity_to_dict(entity)
            category = type_mapping.get(entity.entity_type)
            if category:
                organized[category].append(entity_dict)
            else:
                category_2 = type_mapping_2.get(entity.entity_type)
                if category_2:
                    organized[category_2].append(entity_dict)

        return organized

    def _entity_to_dict(self, entity: Entity) -> Dict[str, Any]:
        """Convert Entity to dictionary"""

        result = {"text": entity.text, "confidence": entity.confidence}

        if entity.role:
            result["role"] = entity.role
        if entity.context:
            result["context"] = entity.context
        if entity.normalized:
            result["normalized"] = entity.normalized
        if entity.metadata:
            result["metadata"] = entity.metadata

        return result

    def _count_sections(self, structure: Any) -> int:
        """Count total sections"""
        if isinstance(structure, Section):
            count = 1
            for subsection in structure.subsections:
                count += self._count_sections(subsection)
            return count
        return 0
