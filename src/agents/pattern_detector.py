import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from src.clients import LLMClient


@dataclass
class PatternDetectorConfig:
    confidence_threshold: float = 0.8
    llm_model: str = "claude-opus-4-20250514"
    temperature: float = 0


class LLMPatternDetector:
    def __init__(self, config: PatternDetectorConfig):
        self.config = config
        self.llm = self._initialize_llm()

    def detect_patterns(self, parsed_documents: List[Dict]) -> Dict:
        """Main entry point for pattern detection using LLMs"""

        prepared_docs = self._prepare_documents_for_analysis(parsed_documents)

        patterns = {
            "structural_patterns": self._detect_structural_patterns(prepared_docs),
            "content_patterns": self._detect_content_patterns(prepared_docs),
            "variable_patterns": self._detect_variable_patterns(prepared_docs),
            "conditional_patterns": self._detect_conditional_patterns(prepared_docs),
            "legal_language_patterns": self._detect_legal_language_patterns(prepared_docs),
        }

        return self._synthesize_patterns(patterns, parsed_documents)

    def _prepare_documents_for_analysis(self, documents: List[Dict]) -> Dict:
        """Prepare documents in a format optimized for LLM analysis"""

        summaries = []
        for i, doc in enumerate(documents):
            summary = {
                "doc_id": f"DOC_{i+1}",
                "outline": self._create_document_outline(doc),
                "key_sections": self._extract_key_sections(doc),
                "entities": doc["entities"],
                "metadata": doc["metadata"],
            }
            summaries.append(summary)

        return {"summaries": summaries, "full_docs": documents, "doc_count": len(documents)}

    def _detect_structural_patterns(self, prepared_docs: Dict) -> Dict:
        """Use LLM to detect structural patterns across documents"""

        # Create a comparative view of all document structures
        structural_comparison = self._create_structural_comparison(prepared_docs)

        prompt = f"""
        Analyze these {prepared_docs['doc_count']} legal document structures and identify patterns.

        Document Structures:
        {json.dumps(structural_comparison, indent=2)}

        Identify:
        1. What sections appear in ALL documents (required sections)
        2. What sections appear in SOME documents (optional sections)
        3. Section ordering patterns
        4. Section naming variations (same content, different titles)
        5. Hierarchical patterns (which sections have subsections)

        For each pattern, explain:
        - Why you consider it a pattern
        - How consistent it is across documents
        - Any variations or exceptions

        Return as JSON with this structure:
        {{
            "required_sections": [
                {{
                    "canonical_name": "PARTIES",
                    "variations": ["PARTIES", "PARTY INFORMATION"],
                    "typical_order": 1,
                    "subsection_pattern": ["PLAINTIFF", "DEFENDANT"]
                }}
            ],
            "optional_sections": [...],
            "ordering_rules": [...],
            "structural_insights": "explanation of patterns found"
        }}
        """

        response = self.llm.complete(prompt)
        return json.loads(response)

    def _detect_content_patterns(self, prepared_docs: Dict) -> Dict:
        """Use LLM to detect content patterns within sections"""

        section_groups = self._group_similar_sections(prepared_docs)

        content_patterns = {}

        for section_name, section_instances in section_groups.items():
            prompt = f"""
            Analyze these {len(section_instances)} instances of the "{section_name}" section from different legal documents.

            Section Contents:
            {json.dumps(section_instances, indent=2)}

            Identify:
            1. Fixed phrases that appear in all/most instances
            2. Variable elements (what changes between documents)
            3. The semantic template/pattern of this section
            4. Required vs optional information
            5. Typical paragraph structure

            Distinguish between:
            - Boilerplate text (exactly the same across documents)
            - Template text (same structure but different values)
            - Variable text (completely different but serving same purpose)

            Return as JSON:
            {{
                "section_name": "{section_name}",
                "semantic_purpose": "what this section accomplishes",
                "fixed_phrases": ["phrase1", "phrase2"],
                "template_sentences": [
                    {{
                        "template": "The [PARTY] was employed as [POSITION] from [DATE]",
                        "frequency": 0.8,
                        "variables": ["PARTY", "POSITION", "DATE"]
                    }}
                ],
                "required_information": ["list of must-have info"],
                "optional_information": ["list of optional info"],
                "paragraph_pattern": "description of typical structure"
            }}
            """

            response = self.llm.complete(prompt)
            content_patterns[section_name] = json.loads(response)

        return content_patterns

    def _detect_variable_patterns(self, prepared_docs: Dict) -> Dict:
        """Use LLM to understand how variables are used across documents"""

        # Extract all entities and their contexts
        entity_contexts = self._extract_entity_contexts(prepared_docs)

        prompt = f"""
        Analyze how entities/variables are used across these legal documents.

        Entity Usage Data:
        {json.dumps(entity_contexts, indent=2)}

        For each type of variable (person, date, money, etc.), determine:
        1. Consistent semantic roles (e.g., first date is always hire date)
        2. Contextual patterns (where/how each variable type appears)
        3. Relationships between variables (e.g., dismissal date > hire date)
        4. Validation rules based on usage patterns
        5. Required vs optional variables

        Also identify:
        - Complex variables that are derived from multiple entities
        - Variables that appear in specific combinations
        - Context-dependent variable interpretation

        Return as JSON:
        {{
            "variable_definitions": {{
                "plaintiff_name": {{
                    "type": "person",
                    "semantic_role": "employee filing the suit",
                    "typical_context": ["appears in PARTIES section", "referenced throughout"],
                    "extraction_hints": ["person with role=plaintiff", "first person mentioned"],
                    "required": true
                }}
            }},
            "variable_relationships": [
                {{
                    "type": "temporal",
                    "rule": "dismissal_date must be after hire_date",
                    "confidence": 1.0
                }}
            ],
            "context_patterns": {{
                "date_interpretation": "rules for interpreting which date is which"
            }}
        }}
        """

        response = self.llm.complete(prompt)
        return json.loads(response)

    def _detect_conditional_patterns(self, prepared_docs: Dict) -> Dict:
        """Use LLM to detect conditional logic in document structures"""

        presence_matrix = self._create_presence_matrix(prepared_docs)
        doc_contexts = self._extract_document_contexts(prepared_docs)

        prompt = f"""
        Analyze when certain sections or content appears conditionally across these documents.

        Section Presence Matrix:
        {json.dumps(presence_matrix, indent=2)}

        Document Contexts:
        {json.dumps(doc_contexts, indent=2)}

        Identify:
        1. Sections that appear only under certain conditions
        2. Content variations based on case characteristics
        3. Implicit rules (e.g., "witness list appears when performance is questioned")
        4. Correlations between document features

        Look for patterns like:
        - If X is mentioned, then Y section appears
        - When case involves Z, different legal grounds are cited
        - Certain sections appear together

        Return as JSON:
        {{
            "conditional_sections": [
                {{
                    "section": "DISCRIMINATION_CLAIMS",
                    "condition": "case involves discrimination allegation",
                    "indicators": ["mention of age/gender/race", "reference to AGG"],
                    "confidence": 0.9
                }}
            ],
            "content_variations": [...],
            "correlation_rules": [...],
            "conditional_insights": "explanation of patterns"
        }}
        """

        response = self.llm.complete(prompt)
        return json.loads(response)

    def _detect_legal_language_patterns(self, prepared_docs: Dict) -> Dict:
        """Use LLM to understand legal language patterns and style"""

        legal_content = self._extract_legal_language(prepared_docs)

        prompt = f"""
        Analyze the legal language patterns across these dismissal protection suits.

        Legal Content Samples:
        {json.dumps(legal_content, indent=2)}

        Identify:
        1. Standard legal phrases and their variations
        2. Argumentation patterns
        3. Citation styles and patterns
        4. Formal language requirements
        5. Jurisdiction-specific language

        Pay special attention to:
        - How legal grounds are presented
        - Standard ways of referencing laws
        - Formal phrases that must be preserved exactly
        - Style variations that don't affect legal meaning

        Return as JSON:
        {{
            "standard_legal_phrases": {{
                "dismissal_protection": [
                    "lacks urgent operational necessity",
                    "failed to consider social selection criteria"
                ],
                "procedural": [
                    "violated notification requirements"
                ]
            }},
            "argumentation_templates": [
                {{
                    "type": "violation_claim",
                    "pattern": "The [ACTION] violates [LAW] because [REASON]",
                    "examples": ["..."]
                }}
            ],
            "citation_patterns": {{
                "german_law": "§ [NUMBER] [ABBREVIATION]",
                "examples": ["§ 1 KSchG", "§ 622 BGB"]
            }},
            "style_guide": {{
                "formality_level": "highly formal",
                "tense": "past for facts, present for claims",
                "voice": "mix of active and passive"
            }}
        }}
        """

        response = self.llm.complete(prompt)
        return json.loads(response)

    def _synthesize_patterns(self, patterns: Dict, original_docs: List[Dict]) -> Dict:
        """Synthesize all detected patterns into a final specification"""

        # use LLM to create a coherent pattern specification
        prompt = f"""
        Synthesize these detected patterns into a comprehensive pattern specification for legal document generation.

        Detected Patterns:
        {json.dumps(patterns, indent=2)}

        Create a unified pattern specification that:
        1. Resolves any conflicts between different pattern types
        2. Provides clear rules for document generation
        3. Includes confidence scores based on consistency
        4. Identifies edge cases and exceptions
        5. Creates a practical template structure

        The specification should be immediately usable by a Template Builder agent.

        Return as JSON:
        {{
            "pattern_id": "dismissal_protection_v1",
            "metadata": {{...}},
            "structural_template": {{...}},
            "content_templates": {{...}},
            "variable_system": {{...}},
            "conditional_logic": {{...}},
            "legal_language_bank": {{...}},
            "generation_rules": {{...}},
            "confidence_assessment": {{...}}
        }}
        """

        response = self.llm.complete(prompt)
        final_spec = json.loads(response)

        final_spec["statistics"] = self._calculate_pattern_statistics(patterns, original_docs)
        final_spec["validation_rules"] = self._generate_validation_rules(patterns)

        return final_spec

    def _create_document_outline(self, doc: Dict) -> List[Dict]:
        """Create a hierarchical outline of document structure"""
        outline = []

        def process_section(section, level=0):
            section_outline = {
                "title": section["title"],
                "level": level,
                "has_content": bool(section.get("content", {}).get("raw_text")),
                "subsections": [],
            }

            if "subsections" in section:
                for subsection in section["subsections"]:
                    section_outline["subsections"].append(process_section(subsection, level + 1))

            return section_outline

        for section in doc["structure"]["sections"]:
            outline.append(process_section(section))

        return outline

    def _extract_key_sections(self, doc: Dict) -> Dict:
        """Extract first paragraph of each major section"""
        key_sections = {}

        for section in doc["structure"]["sections"]:
            if section["title"] in ["PARTIES", "STATEMENT OF CLAIM", "RELIEF SOUGHT"]:
                content = section.get("content", {}).get("raw_text", "")
                # Get first 200 characters or first paragraph
                first_para = content.split("\n\n")[0] if content else ""
                key_sections[section["title"]] = first_para[:200]

        return key_sections

    def _group_similar_sections(self, prepared_docs: Dict) -> Dict:
        """Group sections with similar names across documents"""
        section_groups = defaultdict(list)

        for doc_summary in prepared_docs["summaries"]:
            doc_id = doc_summary["doc_id"]

            full_doc = prepared_docs["full_docs"][int(doc_id.split("_")[1]) - 1]

            for section in full_doc["structure"]["sections"]:
                normalized_name = self._normalize_section_name(section["title"])

                section_groups[normalized_name].append(
                    {
                        "doc_id": doc_id,
                        "title": section["title"],
                        "content": section.get("content", {}).get("raw_text", "")[:500],  # First 500 chars
                        "subsections": [s["title"] for s in section.get("subsections", [])],
                    }
                )

        return dict(section_groups)

    def _normalize_section_name(self, name: str) -> str:
        """Normalize section names for grouping"""
        # Simple normalization - in practice, might use LLM for this too
        mappings = {
            "STATEMENT OF CLAIM": "CLAIMS",
            "FACTS": "CLAIMS",
            "FACTUAL BACKGROUND": "CLAIMS",
            "RELIEF SOUGHT": "RELIEF",
            "PRAYER FOR RELIEF": "RELIEF",
        }

        name_upper = name.upper()
        return mappings.get(name_upper, name_upper)

    def _initialize_llm(self):
        """Initialize LLM client using the shared LLMClient class"""
        return LLMClient(temperature=self.config.temperature)


if __name__ == "__main__":
    config = PatternDetectorConfig(confidence_threshold=0.8, llm_model="claude-4-opus")

    detector = LLMPatternDetector(config)
    parsed_documents = [...]

    pattern_spec = detector.detect_patterns(parsed_documents)
    print(json.dumps(pattern_spec, indent=2))
