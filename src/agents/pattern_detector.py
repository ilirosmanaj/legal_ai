import json
from collections import defaultdict
from typing import Dict, List

from src.clients import LLMClient
from src.logger import logger


class PatternDetector:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.log_raw_response = True

    def detect_patterns(self, parsed_documents: List[Dict], raw_documents: List[str]) -> Dict:
        """Main entry point for pattern detection"""

        if len(parsed_documents) != len(raw_documents):
            raise ValueError("Number of parsed and raw documents must match")

        logger.info(f"Starting pattern detection on {len(parsed_documents)} documents")

        prepared_docs = self._prepare_documents_for_analysis(parsed_documents, raw_documents)

        patterns = {
            "structural_patterns": self._detect_structural_patterns(prepared_docs),
            "content_patterns": self._detect_content_patterns(prepared_docs),
            "variable_patterns": self._detect_variable_patterns(prepared_docs),
            "conditional_patterns": self._detect_conditional_patterns(prepared_docs),
            "legal_language_patterns": self._detect_legal_language_patterns(prepared_docs),
            "style_patterns": self._detect_style_patterns(prepared_docs),
        }

        cross_references = self._analyze_cross_references(prepared_docs)
        patterns["cross_references"] = cross_references

        return self._synthesize_patterns(patterns, prepared_docs)

    def _prepare_documents_for_analysis(self, parsed_documents: List[Dict], raw_documents: List[str]) -> Dict:
        """Prepare documents by combining parsed structure with raw content"""

        prepared = {
            "documents": [],
            "summary": {
                "total_docs": len(parsed_documents),
                "doc_types": defaultdict(int),
                "common_sections": defaultdict(int),
                "avg_length": 0,
            },
        }

        total_length = 0

        for i, (parsed, raw) in enumerate(zip(parsed_documents, raw_documents)):
            doc_entry = {
                "id": f"DOC_{i+1}",
                "parsed": parsed,
                "raw": raw,
                "metadata": {
                    "type": parsed.get("type", "unknown"),
                    "word_count": len(raw.split()),
                    "section_count": self._count_sections(parsed.get("structure", {})),
                    "entity_count": sum(len(v) for v in parsed.get("entities", {}).values()),
                },
            }

            doc_entry["section_mapping"] = self._map_sections_to_raw_text(parsed.get("structure", {}), raw)

            prepared["documents"].append(doc_entry)

            prepared["summary"]["doc_types"][doc_entry["metadata"]["type"]] += 1
            total_length += doc_entry["metadata"]["word_count"]

            for section in self._extract_all_sections(parsed.get("structure", {})):
                prepared["summary"]["common_sections"][section["title"]] += 1

        prepared["summary"]["avg_length"] = total_length / len(parsed_documents)

        return prepared

    def _map_sections_to_raw_text(self, structure: Dict, raw_text: str) -> Dict[str, str]:
        """Map each section to its corresponding raw text"""

        section_map = {}

        def extract_section_text(section: Dict, text: str) -> str:
            """Extract raw text for a specific section"""
            title = section.get("title", "")
            content = section.get("content", "")

            lines = text.split("\n")
            section_lines = []
            in_section = False

            for line in lines:
                if title in line and line.strip().startswith("#"):
                    in_section = True
                    continue
                elif in_section and line.strip().startswith("#"):
                    break
                elif in_section:
                    section_lines.append(line)

            return "\n".join(section_lines).strip() or content

        def process_section(section: Dict, prefix: str = ""):
            section_id = prefix + section.get("title", "Unknown")
            section_map[section_id] = extract_section_text(section, raw_text)

            for subsection in section.get("subsections", []):
                process_section(subsection, section_id + "/")

        process_section(structure)
        return section_map

    def _detect_structural_patterns(self, prepared_docs: Dict) -> Dict:
        """Detect structural patterns using both parsed structure and raw formatting"""

        all_structures = []
        for doc in prepared_docs["documents"]:
            structure_summary = self._summarize_structure(doc["parsed"]["structure"])
            structure_summary["raw_formatting"] = self._analyze_raw_formatting(doc["raw"])
            all_structures.append(structure_summary)

        prompt = f"""
        Analyze document structures to identify patterns across {len(all_structures)} legal documents.

        Document structures with formatting analysis:
        {json.dumps(all_structures, indent=2)}

        Consider both:
        1. Logical structure (sections, subsections, hierarchy)
        2. Physical formatting (headers, spacing, numbering styles)

        Identify:
        - Required vs optional sections
        - Section ordering patterns
        - Naming variations for same logical sections
        - Formatting conventions (how sections are marked)
        - Hierarchical patterns
        - Section with case title should hae the actual title in the canonical name, not "CASE TITLE"

        Return as JSON:
        {{
            "required_sections": [
                {{
                    "canonical_name": "PARTIES",
                    "variations": ["PARTIES", "PARTY INFORMATION"],
                    "formatting_patterns": ["## PARTIES", "**PARTIES**"],
                    "typical_order": 1,
                    "occurrence_rate": 1.0,
                    "subsection_pattern": {{
                        "common_subsections": ["PLAINTIFF", "DEFENDANT"],
                        "subsection_formatting": "### or **bold**"
                    }}
                }}
            ],
            "optional_sections": [...],
            "ordering_rules": {{
                "strict_order": true/false,
                "typical_sequence": ["PARTIES", "CLAIM", "RELIEF"],
                "variations_allowed": [...]
            }},
            "formatting_conventions": {{
                "section_markers": "## for main sections",
                "subsection_markers": "### or bold text",
                "metadata_format": "**Key:** Value"
            }},
            "structural_insights": "Overall patterns observed"
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_detect_structural_patterns]: {response}")

        return json.loads(response)

    def _detect_content_patterns(self, prepared_docs: Dict) -> Dict:
        """Detect content patterns using raw text for exact phrasing"""

        section_groups = defaultdict(list)

        for doc in prepared_docs["documents"]:
            for section_path, raw_content in doc["section_mapping"].items():
                if raw_content:  # Only include non-empty sections
                    section_name = section_path.split("/")[-1]  # Get last part
                    section_groups[section_name].append(
                        {
                            "doc_id": doc["id"],
                            "raw_content": raw_content[:1000],  # Limit for LLM
                            "entities": self._extract_section_entities(doc["parsed"]["entities"]),
                        }
                    )

        content_patterns = {}

        for section_name, instances in section_groups.items():
            if len(instances) < 2:  # Need at least 2 instances to find patterns
                continue

            prompt = f"""
            Analyze {len(instances)} instances of "{section_name}" section to find content patterns.

            Section instances with their raw text:
            {json.dumps(instances, indent=2)}

            Identify:
            1. Fixed phrases that appear verbatim in most/all instances
            2. Template sentences (same structure, different values)
            3. Variable elements and their patterns
            4. Paragraph/sentence organization patterns
            5. Legal language that must be preserved exactly
            6. Style variations that don't affect meaning

            Focus on the EXACT wording and phrasing used.

            Return as JSON:
            {{
                "section_name": "{section_name}",
                "fixed_phrases": [
                    {{
                        "phrase": "exact phrase that appears",
                        "frequency": 0.9,
                        "importance": "critical/standard/optional"
                    }}
                ],
                "template_sentences": [
                    {{
                        "template": "The [PARTY] was employed by [COMPANY] as [POSITION]",
                        "frequency": 0.8,
                        "variations": ["alternate phrasings"],
                        "variables": ["PARTY", "COMPANY", "POSITION"]
                    }}
                ],
                "paragraph_patterns": {{
                    "typical_structure": ["opening statement", "details", "conclusion"],
                    "average_length": "2-3 sentences"
                }},
                "legal_requirements": ["phrases that must appear exactly"],
                "style_characteristics": {{
                    "formality": "highly formal",
                    "voice": "passive/active",
                    "tense": "past/present"
                }}
            }}
            """

            response = self.llm.complete(prompt)
            if self.log_raw_response:
                logger.info(f"Raw response [_detect_content_patterns]: {response}")

            content_patterns[section_name] = json.loads(response)

        return content_patterns

    def _detect_variable_patterns(self, prepared_docs: Dict) -> Dict:
        """Detect how variables are used across documents"""

        entity_contexts = []

        for doc in prepared_docs["documents"]:
            entities = doc["parsed"]["entities"]

            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    raw_context = self._extract_raw_context(entity["text"], doc["raw"], window=100)

                    entity_contexts.append(
                        {
                            "doc_id": doc["id"],
                            "entity_type": entity_type,
                            "text": entity["text"],
                            "normalized": entity.get("normalized"),
                            "role": entity.get("role"),
                            "parsed_context": entity.get("context"),
                            "raw_context": raw_context,
                            "section": "UNKNOWN",
                        }
                    )

        prompt = f"""
        Analyze how variables/entities are used across these legal documents.

        Entity usage data with raw context:
        {json.dumps(entity_contexts, indent=2)}  # Limit for LLM

        For each variable type, determine:
        1. Consistent patterns in how they appear in text
        2. Typical surrounding words/phrases
        3. Relationships between variables
        4. Extraction patterns that would work across documents
        5. Validation rules based on usage

        Pay special attention to:
        - How dates are introduced (hire date vs dismissal date)
        - How parties are referenced after first mention
        - Standard formats for each variable type

        Return as valid loadable JSON:
        {{
            "variable_definitions": {{
                "plaintiff_name": {{
                    "type": "person",
                    "common_patterns": [
                        "The Plaintiff, [NAME],",
                        "[NAME] (hereinafter 'Plaintiff')"
                    ],
                    "extraction_hints": {{
                        "look_for": ["Plaintiff", "Kläger"],
                        "typically_in_section": "PARTIES",
                        "format_variations": ["First Last", "Title First Last"]
                    }},
                    "subsequent_references": ["Plaintiff", "Employee", "Claimant"],
                    "validation": {{
                        "must_be_person_name": true,
                        "appears_multiple_times": true
                    }}
                }}
            }},
            "variable_relationships": [
                {{
                    "type": "employment",
                    "pattern": "[EMPLOYEE] was employed by [EMPLOYER] as [POSITION]",
                    "required_variables": ["employee", "employer", "position"],
                    "optional_variables": ["department", "location"]
                }}
            ],
            "extraction_rules": {{
                "date_disambiguation": "How to tell which date is which",
                "entity_resolution": "How to link pronouns to entities"
            }}
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_detect_variable_patterns]: {response}")

        return json.loads(response)

    def _detect_legal_language_patterns(self, prepared_docs: Dict) -> Dict:
        """Analyze legal language patterns from raw documents"""

        # Extract legal phrases and arguments from raw text
        legal_content_samples = []

        for doc in prepared_docs["documents"]:
            # Focus on sections likely to contain legal language
            legal_sections = [
                "STATEMENT OF CLAIM",
                "CLAIMS",
                "LEGAL GROUNDS",
                "LEGAL ARGUMENTS",
                "RELIEF SOUGHT",
            ]

            for section_path, content in doc["section_mapping"].items():
                if any(legal_sec in section_path.upper() for legal_sec in legal_sections):
                    legal_content_samples.append(
                        {
                            "doc_id": doc["id"],
                            "section": section_path,
                            "content": content[:1500],
                            "legal_refs": [ref["text"] for ref in doc["parsed"]["entities"].get("legal_refs", [])],
                        }
                    )

        prompt = f"""
        Analyze legal language patterns across these documents using the EXACT text.

        Legal content samples:
        {json.dumps(legal_content_samples, indent=2)}

        Identify:
        1. Standard legal phrases and their exact wording
        2. How legal arguments are structured
        3. Citation patterns and formats
        4. Transitional phrases used in legal reasoning
        5. Formal language requirements
        6. Jurisdiction-specific language

        Pay attention to:
        - Exact wording that appears multiple times
        - Variations that mean the same thing legally
        - Required legal formulations

        Return as JSON:
        {{
            "standard_phrases": {{
                "dismissal_related": [
                    {{
                        "phrase": "lacks urgent operational necessity",
                        "variations": ["lacks the required urgent operational necessity"],
                        "legal_significance": "key phrase for KSchG violation",
                        "must_appear_exactly": false
                    }}
                ],
                "procedural": [...],
                "remedial": [...]
            }},
            "argument_patterns": [
                {{
                    "type": "violation_claim",
                    "structure": "The [ACTION] violates [LAW] because [REASON]",
                    "example": "The dismissal violates § 1 KSchG because...",
                    "required_elements": ["action", "law", "reasoning"]
                }}
            ],
            "citation_formats": {{
                "german_law": {{
                    "format": "§ [NUMBER] [LAW_CODE]",
                    "examples": ["§ 1 KSchG", "§ 622 BGB"],
                    "spacing_rules": "space after §"
                }}
            }},
            "transition_phrases": [
                "Furthermore,",
                "In addition,",
                "Moreover,"
            ],
            "formal_requirements": {{
                "person_references": "Use formal titles (Dr., Prof.)",
                "date_format": "European (DD.MM.YYYY) or written out",
                "monetary_format": "€ symbol before amount"
            }}
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_detect_legal_language_patterns]: {response}")

        return json.loads(response)

    def _detect_style_patterns(self, prepared_docs: Dict) -> Dict:
        """Detect writing style patterns from raw documents"""

        # Sample different parts of raw documents
        style_samples = []

        for doc in prepared_docs["documents"]:
            # Get samples from different sections
            intro_sample = doc["raw"][:500]
            middle_sample = doc["raw"][len(doc["raw"]) // 2 : len(doc["raw"]) // 2 + 500]

            style_samples.append(
                {
                    "doc_id": doc["id"],
                    "intro": intro_sample,
                    "middle": middle_sample,
                    "metadata_section": self._extract_metadata_section(doc["raw"]),
                }
            )

        prompt = f"""
        Analyze writing style patterns across these legal documents.

        Document samples:
        {json.dumps(style_samples, indent=2)}

        Identify:
        1. Overall tone and formality level
        2. Sentence structure patterns
        3. Paragraph length and organization
        4. Voice preferences (active vs passive)
        5. Formatting conventions
        6. Punctuation and capitalization patterns

        Return as JSON:
        {{
            "general_style": {{
                "formality_level": "highly formal/formal/semi-formal",
                "tone": "objective/adversarial/neutral",
                "voice_preference": {{
                    "primary": "active/passive",
                    "usage_pattern": "when each is used"
                }}
            }},
            "sentence_patterns": {{
                "average_length": "long/medium/short",
                "complexity": "complex with subclauses/simple and direct",
                "opening_patterns": ["The Plaintiff", "It is", "common starts"]
            }},
            "formatting_style": {{
                "headers": "ALL CAPS/Title Case/Sentence case",
                "emphasis": "**bold**/CAPS/italics",
                "lists": "numbered/bulleted/dashed",
                "spacing": "single/double between sections"
            }},
            "language_preferences": {{
                "word_choices": {{"dismiss": "terminate/dismiss/let go"}},
                "technical_terms": "use of Latin/German legal terms"
            }}
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_detect_style_patterns]: {response}")

        return json.loads(response)

    def _detect_conditional_patterns(self, prepared_docs: Dict) -> Dict:
        """Detect conditional patterns using both structure and content"""

        # Create presence matrix
        doc_features = []

        for doc in prepared_docs["documents"]:
            features = {
                "doc_id": doc["id"],
                "doc_type": doc["metadata"]["type"],
                "sections_present": list(doc["section_mapping"].keys()),
                "entities_mentioned": {
                    "has_discrimination": self._check_discrimination_mentions(doc),
                    "has_witnesses": bool(self._find_witness_section(doc)),
                    "has_monetary_claims": len(doc["parsed"]["entities"].get("monetary", [])) > 0,
                    "has_multiple_claims": self._count_claims(doc) > 1,
                },
                "case_characteristics": self._extract_case_characteristics(doc),
            }
            doc_features.append(features)

        prompt = f"""
        Analyze when certain content appears conditionally across documents.

        Document features and characteristics:
        {json.dumps(doc_features, indent=2)}

        Identify:
        1. Sections that only appear under certain conditions
        2. Content that varies based on case type
        3. Correlations between document features
        4. Implicit rules for including/excluding content

        Return as JSON:
        {{
            "conditional_sections": [
                {{
                    "section": "DISCRIMINATION_GROUNDS",
                    "appears_when": {{
                        "condition": "discrimination is alleged",
                        "indicators": ["AGG mentioned", "age/gender/race referenced"],
                        "correlation_strength": 0.95
                    }},
                    "typical_content": "what this section contains",
                    "required_if_condition_met": true
                }}
            ],
            "content_variations": [
                {{
                    "element": "relief sought",
                    "varies_based_on": "dismissal type",
                    "variations": {{
                        "wrongful_termination": ["reinstatement", "back pay"],
                        "constructive_dismissal": ["severance", "reference letter"]
                    }}
                }}
            ],
            "conditional_language": [
                {{
                    "condition": "performance-based dismissal",
                    "then_include": ["performance documentation", "warning references"],
                    "language_changes": "more emphasis on objective criteria"
                }}
            ]
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_detect_conditional_patterns]: {response}")

        return json.loads(response)

    def _analyze_cross_references(self, prepared_docs: Dict) -> Dict:
        """Analyze how documents reference other sections/entities"""

        cross_refs = []

        for doc in prepared_docs["documents"]:
            # Look for cross-references in raw text
            refs = self._extract_cross_references(doc["raw"])
            refs["doc_id"] = doc["id"]
            cross_refs.append(refs)

        prompt = f"""
        Analyze cross-reference patterns in these documents.

        Cross-references found:
        {json.dumps(cross_refs, indent=2)}

        Identify:
        1. How sections reference each other
        2. How entities are referenced after first mention
        3. Standard reference formats
        4. Pronoun usage patterns

        Return as JSON:
        {{
            "section_references": {{
                "patterns": ["as stated in Section", "see above", "as mentioned"],
                "forward_references": ["as detailed below", "see Section X"],
                "reference_style": "formal/informal"
            }},
            "entity_references": {{
                "first_mention": ["The Plaintiff, NAME,", "NAME (hereinafter 'Plaintiff')"],
                "subsequent": ["the Plaintiff", "Plaintiff", "she/he"],
                "company_references": ["the Company", "Defendant", "the Employer"]
            }},
            "pronoun_usage": {{
                "personal_pronouns": "avoided/used sparingly/used normally",
                "demonstratives": ["this", "that", "these"]
            }}
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_analyze_cross_references]: {response}")

        return json.loads(response)

    def _synthesize_patterns(self, patterns: Dict, prepared_docs: Dict) -> Dict:
        """Synthesize all patterns into final specification"""

        # Get document statistics
        stats = {
            "total_documents": len(prepared_docs["documents"]),
            "document_types": dict(prepared_docs["summary"]["doc_types"]),
            "avg_length": prepared_docs["summary"]["avg_length"],
            "common_sections": dict(prepared_docs["summary"]["common_sections"]),
        }

        prompt = f"""
        Synthesize all detected patterns into a comprehensive pattern specification.

        Document statistics:
        {json.dumps(stats, indent=2)}

        All detected patterns:
        {json.dumps(patterns, indent=2)}

        Create a unified specification that:
        1. Resolves any conflicts between patterns
        2. Identifies the most reliable patterns
        3. Provides confidence scores
        4. Creates practical generation rules
        5. Highlights firm-specific preferences

        Return as JSON:
        {{
            "pattern_id": "legal_doc_pattern_v1",
            "metadata": {{
                "document_type": "primary type",
                "sample_size": {stats['total_documents']},
                "confidence": "overall confidence score",
                "generation_date": "ISO date",
                "key_insights": ["main discoveries"]
            }},
            "document_template": {{
                "structure": {{...}},
                "required_sections": [...],
                "optional_sections": [...]
            }},
            "content_patterns": {{
                "by_section": {{...}},
                "fixed_content": {{...}},
                "variable_content": {{...}}
            }},
            "variable_system": {{
                "definitions": {{...}},
                "extraction_rules": {{...}},
                "relationships": [...]
            }},
            "style_guide": {{
                "tone": "...",
                "formatting": {{...}},
                "legal_language": {{...}}
            }},
            "conditional_rules": [...],
            "generation_instructions": {{
                "step_by_step": [...],
                "quality_checks": [...],
                "firm_preferences": {{...}}
            }},
            "confidence_assessment": {{
                "high_confidence_patterns": [...],
                "medium_confidence_patterns": [...],
                "areas_needing_review": [...]
            }}
        }}
        """

        response = self.llm.complete(prompt)
        if self.log_raw_response:
            logger.info(f"Raw response [_synthesize_patterns]: {response}")

        final_spec = json.loads(response)

        # Add computed statistics
        final_spec["statistics"] = self._calculate_pattern_statistics(patterns)

        return final_spec

    def _count_sections(self, structure: Dict) -> int:
        """Count total sections in document structure"""
        count = 1 if structure else 0
        for subsection in structure.get("subsections", []):
            count += self._count_sections(subsection)
        return count

    def _extract_all_sections(self, structure: Dict) -> List[Dict]:
        """Extract all sections as flat list"""
        sections = [structure] if structure else []
        for subsection in structure.get("subsections", []):
            sections.extend(self._extract_all_sections(subsection))
        return sections

    def _summarize_structure(self, structure: Dict) -> Dict:
        """Create summary of document structure"""

        def get_section_info(section: Dict, level: int = 0) -> Dict:
            info = {
                "title": section.get("title", ""),
                "level": level,
                "has_content": bool(section.get("content", "").strip()),
                "subsection_count": len(section.get("subsections", [])),
            }

            if section.get("subsections"):
                info["subsections"] = [get_section_info(sub, level + 1) for sub in section["subsections"]]

            return info

        return get_section_info(structure)

    def _analyze_raw_formatting(self, raw_text: str) -> Dict:
        """Analyze formatting patterns in raw text"""

        lines = raw_text.split("\n")

        formatting = {"header_styles": [], "emphasis_patterns": [], "list_styles": [], "spacing_patterns": []}

        for _, line in enumerate(lines):
            if line.strip().startswith("#"):
                level = len(line.split()[0])
                formatting["header_styles"].append({"level": level, "example": line.strip()[:50]})

            # Check emphasis
            if "**" in line:
                formatting["emphasis_patterns"].append("bold_markdown")
            if "__" in line:
                formatting["emphasis_patterns"].append("underscore_markdown")

            # Check list styles
            if line.strip().startswith(("- ", "* ", "+ ")):
                formatting["list_styles"].append("bullet")
            elif line.strip() and line.strip()[0].isdigit() and "." in line[:5]:
                formatting["list_styles"].append("numbered")

        # Deduplicate
        for key in formatting:
            if isinstance(formatting[key], list) and key != "header_styles":
                formatting[key] = list(set(formatting[key]))

        return formatting

    def _extract_section_entities(self, all_entities: Dict) -> List[Dict]:
        """Extract entities that belong to a specific section"""
        # This is simplified - in production would use character positions
        section_entities = []

        # For now, return a subset of entities
        # In real implementation, would check entity positions against section boundaries
        for entity_type, entities in all_entities.items():
            for entity in entities[:2]:  # Just take first 2 of each type
                section_entities.append(
                    {"type": entity_type, "text": entity.get("text", ""), "role": entity.get("role", "")}
                )

        return section_entities

    def _extract_raw_context(self, entity_text: str, raw_text: str, window: int = 100) -> str:
        """Extract context around entity from raw text"""

        # Find entity in raw text
        pos = raw_text.find(entity_text)
        if pos == -1:
            return ""

        # Extract surrounding context
        start = max(0, pos - window)
        end = min(len(raw_text), pos + len(entity_text) + window)

        context = raw_text[start:end]

        # Clean up
        context = " ".join(context.split())

        # Mark the entity
        context = context.replace(entity_text, f"[{entity_text}]")

        return context

    def _check_discrimination_mentions(self, doc: Dict) -> bool:
        """Check if document mentions discrimination"""

        discrimination_terms = [
            "discrimination",
            "diskriminierung",
            "AGG",
            "age",
            "gender",
            "race",
            "religion",
            "disability",
        ]

        raw_lower = doc["raw"].lower()
        return any(term in raw_lower for term in discrimination_terms)

    def _find_witness_section(self, doc: Dict) -> bool:
        """Check if document has witness section"""

        witness_indicators = ["witness", "zeuge", "testimony"]

        for section_path in doc["section_mapping"]:
            if any(ind in section_path.lower() for ind in witness_indicators):
                return True

        return False

    def _count_claims(self, doc: Dict) -> int:
        """Count number of claims in document"""

        # Look for claim indicators
        claim_sections = [
            path for path in doc["section_mapping"] if "claim" in path.lower() or "ground" in path.lower()
        ]

        return len(claim_sections)

    def _extract_case_characteristics(self, doc: Dict) -> Dict:
        """Extract characteristics of the case"""

        characteristics = {"dismissal_type": "unknown", "has_monetary_claim": False, "urgency_indicated": False}

        raw_lower = doc["raw"].lower()

        # Determine dismissal type
        if "operational" in raw_lower or "restructuring" in raw_lower:
            characteristics["dismissal_type"] = "operational"
        elif "performance" in raw_lower or "misconduct" in raw_lower:
            characteristics["dismissal_type"] = "performance"
        elif "discrimination" in raw_lower:
            characteristics["dismissal_type"] = "discrimination"

        # Check for monetary claims
        characteristics["has_monetary_claim"] = bool(doc["parsed"]["entities"].get("monetary"))

        # Check urgency
        characteristics["urgency_indicated"] = "urgent" in raw_lower or "immediate" in raw_lower

        return characteristics

    def _extract_metadata_section(self, raw_text: str) -> str:
        """Extract metadata section from raw document"""

        lines = raw_text.split("\n")[:20]  # First 20 lines

        metadata_lines = []
        for line in lines:
            if "**" in line and ":" in line:  # Typical metadata format
                metadata_lines.append(line)

        return "\n".join(metadata_lines)

    def _extract_cross_references(self, raw_text: str) -> Dict:
        """Extract cross-references from document"""

        references = {"section_refs": [], "above_refs": 0, "below_refs": 0, "see_refs": 0}

        # Simple pattern matching
        if "see above" in raw_text.lower():
            references["above_refs"] = raw_text.lower().count("see above")
        if "see below" in raw_text.lower():
            references["below_refs"] = raw_text.lower().count("see below")
        if "see section" in raw_text.lower():
            references["see_refs"] = raw_text.lower().count("see section")

        return references

    def _calculate_pattern_statistics(self, patterns: Dict) -> Dict:
        """Calculate statistics about patterns found"""

        stats = {"total_patterns_found": 0, "high_confidence_patterns": 0, "coverage": {}}

        for _, pattern_data in patterns.items():
            if isinstance(pattern_data, dict):
                stats["total_patterns_found"] += len(pattern_data)

                # Count high confidence patterns
                # (Would need actual confidence scores from pattern data)
                stats["high_confidence_patterns"] += len(pattern_data) // 2

        return stats

    def _initialize_llm(self):
        """Initialize LLM client using the shared LLMClient class"""
        return LLMClient()
