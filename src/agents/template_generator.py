import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.clients import LLMClient

logger = logging.getLogger(__name__)


class VariableType(Enum):
    TEXT = "text"
    DATE = "date"
    MONEY = "money"
    NUMBER = "number"
    CHOICE = "choice"
    BOOLEAN = "boolean"
    LIST = "list"


class ConditionType(Enum):
    ALWAYS = "always"
    IF_PROVIDED = "if_provided"
    IF_TRUE = "if_true"
    IF_EQUALS = "if_equals"
    IF_CONTAINS = "if_contains"
    IF_GREATER = "if_greater"
    IF_LESS = "if_less"
    CUSTOM = "custom"


@dataclass
class Variable:
    name: str
    type: VariableType
    required: bool = True
    default: Optional[Any] = None
    validation: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    ui_hints: Optional[Dict[str, Any]] = None
    extraction_hints: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type.value,
            "required": self.required,
            "default": self.default,
            "validation": self.validation,
            "description": self.description,
            "ui_hints": self.ui_hints,
            "extraction_hints": self.extraction_hints,
            "dependencies": self.dependencies,
        }


@dataclass
class TemplateSection:
    id: str
    title: str
    template_content: str
    variables: List[str]
    required: bool = True
    order: int = 0
    conditions: Optional[List[Dict[str, Any]]] = None
    subsections: Optional[List["TemplateSection"]] = None
    generation_instructions: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []
        if self.conditions is None:
            self.conditions = []


class TemplateBuilder:
    """
    Creates executable templates from pattern specifications.
    The output template can generate new documents by filling in variables.
    """

    def __init__(self):
        self.llm = self._initialize_llm()
        self.log_raw_response = True
        self.template_version = "1.0"

    def build_template(self, pattern_spec: Dict, parsed_documents: List[Dict], raw_documents: List[str]) -> Dict:
        """
        Build an executable template from patterns.

        Args:
            pattern_spec: Output from PatternDetector
            parsed_documents: Parsed document structures
            raw_documents: Original document texts

        Returns:
            Executable template specification
        """

        logger.info("Starting template building process")

        analysis = self._analyze_pattern_spec(pattern_spec)

        variable_system = self._build_variable_system(pattern_spec, parsed_documents)

        template_structure = self._create_template_structure(pattern_spec, analysis)

        section_templates = self._generate_section_templates(pattern_spec, raw_documents, variable_system)

        conditional_logic = self._build_conditional_logic(pattern_spec)

        generation_instructions = self._create_generation_instructions(
            pattern_spec, template_structure, variable_system
        )

        style_guide = self._extract_style_guide(pattern_spec, raw_documents)

        validation_rules = self._build_validation_rules(pattern_spec, variable_system, template_structure)

        executable_template = self._compile_executable_template(
            template_structure,
            section_templates,
            variable_system,
            conditional_logic,
            generation_instructions,
            style_guide,
            validation_rules,
            pattern_spec,
        )

        logger.info("Template building completed")
        return executable_template

    def _analyze_pattern_spec(self, pattern_spec: Dict) -> Dict:
        """Analyze pattern specification to understand requirements"""

        analysis = {
            "document_type": pattern_spec["metadata"].get("document_type", "legal_document"),
            "complexity": self._assess_complexity(pattern_spec),
            "variable_count": len(pattern_spec.get("variable_system", {}).get("definitions", {})),
            "conditional_sections": len(pattern_spec.get("conditional_rules", [])),
            "required_sections": [],
            "optional_sections": [],
            "generation_challenges": [],
        }

        doc_template = pattern_spec.get("document_template", {})
        for section in doc_template.get("required_sections", []):
            analysis["required_sections"].append(section)

        for section in doc_template.get("optional_sections", []):
            analysis["optional_sections"].append(section)

        if analysis["conditional_sections"] > 3:
            analysis["generation_challenges"].append("complex_conditionals")

        if analysis["variable_count"] > 20:
            analysis["generation_challenges"].append("many_variables")

        return analysis

    def _build_variable_system(self, pattern_spec: Dict, parsed_documents: List[Dict]) -> Dict:
        """Build comprehensive variable system"""

        variables = {}
        variable_definitions = pattern_spec.get("variable_system", {}).get("definitions", {})

        for var_name, var_spec in variable_definitions.items():
            variable = Variable(
                name=var_name,
                type=self._determine_variable_type(var_spec),
                required=var_spec.get("required", True),
                description=var_spec.get("description", f"Enter {var_name}"),
                validation=self._create_validation_rules(var_name, var_spec),
                ui_hints=self._create_ui_hints(var_name, var_spec),
                extraction_hints=var_spec.get("extraction_hints", {}),
            )

            variable.ui_hints["examples"] = self._extract_variable_examples(var_name, parsed_documents, var_spec)

            variables[var_name] = variable

        relationships = pattern_spec.get("variable_system", {}).get("relationships", [])
        self._add_variable_dependencies(variables, relationships)

        return {
            "variables": {name: var.to_dict() for name, var in variables.items()},
            "relationships": relationships,
            "input_order": self._determine_input_order(variables, relationships),
            "validation_order": self._determine_validation_order(variables),
        }

    def _create_template_structure(self, pattern_spec: Dict, analysis: Dict) -> Dict:
        """Create the hierarchical template structure"""

        structure = {"document_type": analysis["document_type"], "version": self.template_version, "sections": []}

        doc_template = pattern_spec.get("document_template", {})
        section_order = 1

        for section_spec in doc_template.get("required_sections", []):
            template_section = TemplateSection(
                id=self._generate_section_id(section_spec),
                title=section_spec,
                template_content="",  # Will be filled later
                variables=[],  # Will be extracted later
                required=True,
                order=section_order,
            )

            # Add subsections if specified
            if "subsections" in section_spec:
                for subsection in section_spec["subsections"]:
                    template_section.subsections.append(
                        TemplateSection(
                            id=self._generate_section_id(subsection),
                            title=subsection,
                            template_content="",
                            variables=[],
                            required=True,
                            order=section_order + 0.1,
                        )
                    )

            structure["sections"].append(template_section)
            section_order += 1

        # Add optional sections
        for section_spec in doc_template.get("optional_sections", []):
            template_section = TemplateSection(
                id=self._generate_section_id(section_spec),
                title=section_spec,
                template_content="",
                variables=[],
                required=False,
                order=section_order,
                conditions=self._extract_section_conditions(section_spec),
            )

            structure["sections"].append(template_section)
            section_order += 1

        return structure

    def _generate_section_templates(
        self, pattern_spec: Dict, raw_documents: List[str], variable_system: Dict
    ) -> Dict[str, str]:
        """Generate template content for each section"""

        section_templates = {}
        content_patterns = pattern_spec.get("content_patterns", {}).get("by_section", {})

        for section_name, section_pattern in content_patterns.items():
            section_name = section_name.upper().replace(" ", "_")
            section_examples = self._extract_section_examples(section_name, raw_documents)

            template_content = self._generate_section_template(
                section_name, section_pattern, section_examples, variable_system
            )

            section_templates[section_name] = template_content

        return section_templates

    def _generate_section_template(
        self, section_name: str, section_pattern: Dict, examples: List[str], variable_system: Dict
    ) -> str:
        """Generate template for a specific section using LLM"""

        fixed_phrases = section_pattern.get("fixed_phrases", [])
        template_sentences = section_pattern.get("template_sentences", [])

        prompt = f"""
        Create a template for the "{section_name}" section of a legal document.

        Section pattern information:
        - Fixed phrases that must appear: {json.dumps(fixed_phrases, indent=2)}
        - Template sentences: {json.dumps(template_sentences, indent=2)}
        - Style characteristics: {json.dumps(section_pattern.get('style_characteristics', {}), indent=2)}

        Example content from real documents:
        {'\n------\n'.join(examples)}

        Available variables:
        {json.dumps(list(variable_system['variables'].keys()), indent=2)}

        Create a template that:
        1. Uses {{{{VARIABLE_NAME}}}} for simple variable substitution
        2. Preserves all critical legal language exactly
        3. Uses [[GENERATE: instruction]] for complex generated content
        4. Uses <<IF condition>>content<</IF>> for conditional content
        5. Maintains the style and tone of the examples

        The template should be ready to use - when variables are filled in, it should produce
        a complete, legally valid section.

        Return ONLY the template content, no explanations.
        """

        template = self.llm.complete(prompt)

        template = self._post_process_template(template, section_pattern)

        return template

    def _post_process_template(self, template: str, section_pattern: Dict) -> str:
        """Clean and validate generated template"""

        required_phrases = section_pattern.get("fixed_phrases", [])

        for phrase in required_phrases:
            if phrase not in template:
                template = f"{template}\n\n{phrase}"

        template = self._validate_variable_syntax(template)

        template = self._clean_template_formatting(template)

        return template

    def _build_conditional_logic(self, pattern_spec: Dict) -> Dict:
        """Build conditional logic system"""

        conditional_rules = pattern_spec.get("conditional_rules", [])

        logic_system = {"rules": [], "conditions": {}, "actions": {}, "evaluation_order": []}

        for rule in conditional_rules:
            condition_id = f"condition_{len(logic_system['conditions'])}"
            condition = {
                "id": condition_id,
                "type": self._determine_condition_type(rule),
                "expression": rule.get("condition", ""),
                "evaluation": self._create_condition_evaluator(rule),
            }
            logic_system["conditions"][condition_id] = condition

            action_id = f"action_{len(logic_system['actions'])}"
            action = {
                "id": action_id,
                "type": rule.get("action_type", "include_section"),
                "target": rule.get("target"),
                "parameters": rule.get("parameters", {}),
            }
            logic_system["actions"][action_id] = action

            # Create rule
            rule_entry = {
                "id": f"rule_{len(logic_system['rules'])}",
                "condition_id": condition_id,
                "action_id": action_id,
                "priority": rule.get("priority", 100),
            }
            logic_system["rules"].append(rule_entry)

        logic_system["evaluation_order"] = sorted(logic_system["rules"], key=lambda r: r["priority"])

        return logic_system

    def _create_generation_instructions(
        self, pattern_spec: Dict, template_structure: Dict, variable_system: Dict
    ) -> Dict:
        """Create detailed generation instructions"""

        instructions = {
            "generation_flow": [],
            "section_instructions": {},
            "variable_handling": {},
            "style_requirements": {},
            "quality_checks": [],
        }

        for section in template_structure["sections"]:
            step = {
                "step_id": f"gen_{section.id}",
                "section_id": section.id,
                "section_title": section.title,
                "required": section.required,
                "order": section.order,
                "instructions": self._create_section_generation_instructions(section, pattern_spec, variable_system),
            }
            instructions["generation_flow"].append(step)

        instructions["generation_flow"].sort(key=lambda x: x["order"])

        instructions["variable_handling"] = {
            "missing_required": "raise_error",
            "missing_optional": "skip_or_use_default",
            "invalid_format": "attempt_correction_then_error",
            "type_conversion": self._create_type_conversion_rules(),
        }

        style_guide = pattern_spec.get("style_guide", {})
        instructions["style_requirements"] = {
            "tone": style_guide.get("tone", "formal"),
            "voice": style_guide.get("voice_preference", {"primary": "active"}),
            "formatting": style_guide.get("formatting_style", {}),
            "legal_language": style_guide.get("legal_language", {}),
        }

        instructions["quality_checks"] = [
            {"check": "required_sections_present", "severity": "error"},
            {"check": "legal_references_valid", "severity": "error"},
            {"check": "variable_consistency", "severity": "warning"},
            {"check": "style_compliance", "severity": "warning"},
        ]

        return instructions

    def _extract_style_guide(self, pattern_spec: Dict, raw_documents: List[str]) -> Dict:
        """Extract comprehensive style guide"""

        style_patterns = pattern_spec.get("style_guide", {})

        style_guide = {
            "general": {
                "formality_level": style_patterns.get("general_style", {}).get("formality_level", "formal"),
                "tone": style_patterns.get("general_style", {}).get("tone", "objective"),
                "voice": style_patterns.get("general_style", {}).get("voice_preference", {}),
            },
            "formatting": {
                "date_format": self._extract_date_format(raw_documents),
                "number_format": self._extract_number_format(raw_documents),
                "currency_format": style_patterns.get("formatting", {}).get("monetary_format", "€X,XXX"),
                "section_headers": style_patterns.get("formatting_style", {}).get("headers", "ALL CAPS"),
                "emphasis": style_patterns.get("formatting_style", {}).get("emphasis", "**bold**"),
            },
            "language": {
                "standard_phrases": style_patterns.get("standard_phrases", {}),
                "transition_words": style_patterns.get("transition_phrases", []),
                "legal_citations": style_patterns.get("citation_formats", {}),
                "avoid_phrases": style_patterns.get("avoid", []),
            },
            "structure": {
                "paragraph_length": style_patterns.get("sentence_patterns", {}).get("average_length", "medium"),
                "sentence_complexity": style_patterns.get("sentence_patterns", {}).get("complexity", "complex"),
                "list_style": style_patterns.get("formatting_style", {}).get("lists", "numbered"),
            },
        }

        return style_guide

    def _build_validation_rules(self, pattern_spec: Dict, variable_system: Dict, template_structure: Dict) -> Dict:
        """Build comprehensive validation rules"""

        validation = {
            "variable_validation": {},
            "structural_validation": [],
            "content_validation": [],
            "relationship_validation": [],
            "legal_validation": [],
        }

        for var_name, var_def in variable_system["variables"].items():
            validation["variable_validation"][var_name] = var_def.get("validation", {})

        validation["structural_validation"] = [
            {
                "rule": "all_required_sections_present",
                "sections": [s.title for s in template_structure["sections"] if s.required],
                "severity": "error",
            },
            {
                "rule": "section_order_maintained",
                "expected_order": [s.title for s in sorted(template_structure["sections"], key=lambda x: x.order)],
                "severity": "warning",
            },
        ]

        content_patterns = pattern_spec.get("content_patterns", {}).get("by_section", {})
        for section_name, section_pattern in content_patterns.items():
            critical_phrases = section_pattern.get("fixed_phrases", [])

            if critical_phrases:
                validation["content_validation"].append(
                    {
                        "section": section_name,
                        "rule": "contains_required_phrases",
                        "phrases": critical_phrases,
                        "severity": "error",
                    }
                )

        for relationship in variable_system.get("relationships", []):
            if relationship["type"] == "temporal":
                validation["relationship_validation"].append(
                    {
                        "rule": "temporal_consistency",
                        "before": relationship["before"],
                        "after": relationship["after"],
                        "severity": "error",
                    }
                )
            elif relationship["type"] == "logical":
                validation["relationship_validation"].append(
                    {"rule": "logical_consistency", "condition": relationship["condition"], "severity": "warning"}
                )

        validation["legal_validation"] = [
            {
                "rule": "valid_legal_references",
                "pattern": pattern_spec.get("legal_language_patterns", {}).get("citation_formats", {}),
                "severity": "error",
            },
            {
                "rule": "required_legal_elements",
                "elements": self._extract_required_legal_elements(pattern_spec),
                "severity": "error",
            },
        ]

        return validation

    def _compile_executable_template(
        self,
        template_structure: Dict,
        section_templates: Dict[str, str],
        variable_system: Dict,
        conditional_logic: Dict,
        generation_instructions: Dict,
        style_guide: Dict,
        validation_rules: Dict,
        pattern_spec: Dict,
    ) -> Dict:
        """Compile all components into executable template"""
        for section in template_structure["sections"]:
            section_key = section.title.upper().replace(" ", "_")
            if section_key in section_templates:
                section.template_content = section_templates[section_key]
                section.variables = self._extract_variables_from_template(section.template_content)

            for subsection in section.subsections:
                subsection_key = f"{section_key}/{subsection.title.upper().replace(' ', '_')}"
                if subsection_key in section_templates:
                    subsection.template_content = section_templates[subsection_key]
                    subsection.variables = self._extract_variables_from_template(subsection.template_content)

        executable_template = {
            "template_id": f"template_{pattern_spec['pattern_id']}_{self.template_version}",
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "version": self.template_version,
                "pattern_id": pattern_spec.get("pattern_id"),
                "document_type": template_structure["document_type"],
                "sample_size": pattern_spec["metadata"].get("sample_size", 0),
                "confidence": pattern_spec["metadata"].get("confidence", 0.0),
            },
            "structure": self._serialize_structure(template_structure),
            "variable_system": variable_system,
            "conditional_logic": conditional_logic,
            "generation_instructions": generation_instructions,
            "style_guide": style_guide,
            "validation_rules": validation_rules,
            "user_interface": {
                "input_fields": self._create_input_fields(variable_system),
                "conditional_fields": self._create_conditional_fields(conditional_logic),
                "sections_toggle": self._create_section_toggles(template_structure),
                "preview_available": True,
            },
            "generation_config": {
                "method": "template_fill_and_generate",
                "llm_required": self._check_if_llm_required(template_structure),
                "post_processing": [
                    "format_dates",
                    "format_currency",
                    "validate_legal_references",
                    "ensure_consistency",
                ],
            },
        }

        return executable_template

    def _assess_complexity(self, pattern_spec: Dict) -> str:
        """Assess template complexity"""

        factors = {
            "variable_count": len(pattern_spec.get("variable_system", {}).get("definitions", {})),
            "conditional_count": len(pattern_spec.get("conditional_rules", [])),
            "section_count": len(pattern_spec.get("document_template", {}).get("required_sections", [])),
        }

        complexity_score = (
            (factors["variable_count"] / 10) + (factors["conditional_count"] / 5) + (factors["section_count"] / 10)
        ) / 3

        if complexity_score < 0.3:
            return "simple"
        elif complexity_score < 0.7:
            return "moderate"
        else:
            return "complex"

    def _determine_variable_type(self, var_spec: Dict) -> VariableType:
        """Determine variable type from specification"""

        type_str = var_spec.get("type", "text").lower()

        type_mapping = {
            "text": VariableType.TEXT,
            "string": VariableType.TEXT,
            "person": VariableType.TEXT,
            "organization": VariableType.TEXT,
            "date": VariableType.DATE,
            "money": VariableType.MONEY,
            "monetary": VariableType.MONEY,
            "currency": VariableType.MONEY,
            "number": VariableType.NUMBER,
            "choice": VariableType.CHOICE,
            "boolean": VariableType.BOOLEAN,
            "list": VariableType.LIST,
        }

        return type_mapping.get(type_str, VariableType.TEXT)

    def _create_validation_rules(self, var_name: str, var_spec: Dict) -> Dict:
        """Create validation rules for a variable"""

        validation = {}

        if var_spec.get("type") == "date":
            validation["format"] = "YYYY-MM-DD"
            if "validation" in var_spec:
                validation.update(var_spec["validation"])

        elif var_spec.get("type") in ["money", "monetary"]:
            validation["min"] = 0
            validation["currency"] = var_spec.get("currency", "EUR")

        elif var_spec.get("type") == "text":
            if "max_length" in var_spec:
                validation["max_length"] = var_spec["max_length"]
            if "pattern" in var_spec:
                validation["pattern"] = var_spec["pattern"]

        if "validation" in var_spec:
            validation.update(var_spec["validation"])

        return validation

    def _create_ui_hints(self, var_name: str, var_spec: Dict) -> Dict:
        """Create UI hints for variable input"""

        ui_hints = {
            "label": self._humanize_variable_name(var_name),
            "placeholder": f"Enter {self._humanize_variable_name(var_name).lower()}",
            "help_text": var_spec.get("description", ""),
            "input_type": self._determine_input_type(var_spec.get("type", "text")),
        }

        if var_spec.get("type") == "choice":
            ui_hints["options"] = var_spec.get("options", [])

        elif var_spec.get("type") == "date":
            ui_hints["date_picker"] = True
            ui_hints["format_hint"] = "DD.MM.YYYY"

        elif var_spec.get("type") in ["money", "monetary"]:
            ui_hints["currency_symbol"] = "€"
            ui_hints["decimal_places"] = 2

        return ui_hints

    def _extract_variable_examples(self, var_name: str, parsed_documents: List[Dict], var_spec: Dict) -> List[str]:
        """Extract examples of variable from parsed documents"""

        examples = []

        entity_mapping = {
            "plaintiff_name": ("persons", "plaintiff"),
            "defendant_name": ("organizations", "defendant"),
            "defendant_org": ("organizations", "defendant"),
            "hire_date": ("dates", "hire_date"),
            "dismissal_date": ("dates", "dismissal_date"),
            "salary": ("monetary", "salary"),
            "monthly_salary": ("monetary", "salary"),
        }

        if var_name in entity_mapping:
            entity_type, role = entity_mapping[var_name]

            for doc in parsed_documents:
                entities = doc.get("entities", {}).get(entity_type, [])
                for entity in entities:
                    if role in str(entity.get("role", "")) or role in str(entity.get("context", "")):
                        if entity.get("text") not in examples:
                            examples.append(entity.get("text"))

        return examples

    def _add_variable_dependencies(self, variables: Dict[str, Variable], relationships: List[Dict]):
        """Add dependencies between variables"""

        for relationship in relationships:
            if relationship["type"] == "temporal":
                # dismissal_date depends on hire_date
                if relationship["after"] in variables and relationship["before"] in variables:
                    variables[relationship["after"]].dependencies.append(relationship["before"])

            elif relationship["type"] == "calculated":
                # Calculated fields depend on source fields
                if "target" in relationship and "sources" in relationship:
                    if relationship["target"] in variables:
                        variables[relationship["target"]].dependencies.extend(relationship["sources"])

    def _determine_input_order(self, variables: Dict[str, Variable], relationships: List[Dict]) -> List[str]:
        """Determine optimal order for variable input"""

        # Topological sort based on dependencies
        ordered = []
        visited = set()

        def visit(var_name: str):
            if var_name in visited:
                return
            visited.add(var_name)

            if var_name in variables:
                for dep in variables[var_name].dependencies:
                    visit(dep)
                ordered.append(var_name)

        for var_name in variables:
            visit(var_name)

        return ordered

    def _determine_validation_order(self, variables: Dict[str, Variable]) -> List[str]:
        """Determine order for validation"""

        return self._determine_input_order(variables, [])

    def _generate_section_id(self, title: str) -> str:
        """Generate clean section ID"""
        return re.sub(r"[^a-zA-Z0-9_]", "_", title.lower())

    def _extract_section_conditions(self, section_spec: Dict) -> List[Dict]:
        """Extract conditions for optional sections"""

        conditions = []

        if "appears_when" in section_spec:
            conditions.append(
                {
                    "type": ConditionType.CUSTOM.value,
                    "expression": section_spec["appears_when"],
                    "description": f"Include when: {section_spec['appears_when']}",
                }
            )

        return conditions

    def _extract_section_examples(self, section_name: str, raw_documents: List[str]) -> List[str]:
        """Extract examples of a section from raw documents"""

        examples = []
        for raw_doc in raw_documents:
            lines = raw_doc.split("\n")
            in_section = False
            section_content = []

            for line in lines:
                if section_name.upper() in line.upper() and line.strip().startswith("#"):
                    in_section = True
                    continue
                elif in_section and line.strip().startswith("#"):
                    break
                elif in_section:
                    section_content.append(line)

            if section_content:
                examples.append("\n".join(section_content).strip())

        return examples

    def _validate_variable_syntax(self, template: str) -> str:
        """Validate and fix variable syntax in template"""

        template = re.sub(r"\{([A-Z_]+)\}", r"{{\1}}", template)
        template = re.sub(r"\[GENERATE: ([^\]]+)\]", r"[[GENERATE: \1]]", template)
        template = re.sub(r"<IF ([^>]+)>", r"<<IF \1>>", template)
        template = re.sub(r"</IF>", r"<</IF>>", template)

        return template

    def _clean_template_formatting(self, template: str) -> str:
        """Clean up template formatting"""

        template = re.sub(r"\n\n\n+", "\n\n", template)
        template = template.strip()

        return template

    def _determine_condition_type(self, rule: Dict) -> ConditionType:
        """Determine condition type from rule"""

        condition = rule.get("condition", "").lower()

        if "if provided" in condition:
            return ConditionType.IF_PROVIDED
        elif "==" in condition or "equals" in condition:
            return ConditionType.IF_EQUALS
        elif "contains" in condition:
            return ConditionType.IF_CONTAINS
        elif ">" in condition:
            return ConditionType.IF_GREATER
        elif "<" in condition:
            return ConditionType.IF_LESS
        else:
            return ConditionType.CUSTOM

    def _create_condition_evaluator(self, rule: Dict) -> Dict:
        """Create evaluator for condition"""

        return {
            "type": "expression",
            "expression": rule.get("condition", ""),
            "variables": self._extract_variables_from_condition(rule.get("condition", "")),
        }

    def _extract_variables_from_condition(self, condition: str) -> List[str]:
        """Extract variable names from condition"""

        variables = re.findall(r"\b([a-z_]+)\b", condition.lower())

        return [v for v in variables if "_" in v or v in ["age", "salary", "date"]]

    def _create_section_generation_instructions(
        self, section: TemplateSection, pattern_spec: Dict, variable_system: Dict
    ) -> Dict:
        """Create generation instructions for a section"""

        content_pattern = pattern_spec.get("content_patterns", {}).get("by_section", {}).get(section.title, {})

        return {
            "approach": "template_based" if section.template_content else "llm_generated",
            "required_elements": content_pattern.get("legal_requirements", []),
            "style_notes": content_pattern.get("style_characteristics", {}),
            "variable_handling": {"missing_optional": "skip_placeholder", "missing_required": "error"},
            "quality_checks": ["verify_legal_language", "check_variable_consistency", "validate_formatting"],
        }

    def _create_type_conversion_rules(self) -> Dict:
        """Create rules for type conversion"""

        return {
            "date": {
                "input_formats": ["DD.MM.YYYY", "YYYY-MM-DD", "Month DD, YYYY"],
                "output_format": "DD.MM.YYYY",
                "conversion": "flexible_date_parser",
            },
            "money": {
                "input_formats": ["1234.56", "1,234.56", "1.234,56"],
                "output_format": "€X,XXX.XX",
                "conversion": "currency_formatter",
            },
            "text": {"capitalization": "preserve", "trimming": "trim_whitespace"},
        }

    def _extract_date_format(self, raw_documents: List[str]) -> str:
        """Extract predominant date format from documents"""

        formats = {"DD.MM.YYYY": 0, "YYYY-MM-DD": 0, "Month DD, YYYY": 0}

        for doc in raw_documents[:10]:
            if re.search(r"\d{1,2}\.\d{1,2}\.\d{4}", doc):
                formats["DD.MM.YYYY"] += 1
            if re.search(r"\d{4}-\d{2}-\d{2}", doc):
                formats["YYYY-MM-DD"] += 1
            if re.search(r"[A-Z][a-z]+ \d{1,2}, \d{4}", doc):
                formats["Month DD, YYYY"] += 1

        return max(formats.items(), key=lambda x: x[1])[0]

    def _extract_number_format(self, raw_documents: List[str]) -> str:
        """Extract number format from documents"""

        for doc in raw_documents[:5]:
            if "€" in doc and "," in doc:
                return "European"  # 1.234,56

        return "US"  # 1,234.56

    def _extract_required_legal_elements(self, pattern_spec: Dict) -> List[str]:
        """Extract required legal elements from pattern"""

        elements = []

        content_patterns = pattern_spec.get("content_patterns", {}).get("by_section", {})
        for section_pattern in content_patterns.values():
            elements.extend(section_pattern.get("legal_requirements", []))

        legal_patterns = pattern_spec.get("legal_language_patterns", {})
        for category in legal_patterns.get("standard_phrases", {}).values():
            for phrase in category:
                if isinstance(phrase, dict) and phrase.get("must_appear_exactly"):
                    elements.append(phrase["phrase"])

        return list(set(elements))

    def _extract_variables_from_template(self, template: str) -> List[str]:
        """Extract variable names from template content"""

        variables = []

        var_matches = re.findall(r"{{([A-Z_]+)}}", template)
        variables.extend(var_matches)

        cond_matches = re.findall(r"<<IF\s+(\w+)", template)
        variables.extend(cond_matches)

        return list(set(variables))

    def _serialize_structure(self, template_structure: Dict) -> Dict:
        """Serialize template structure for storage"""

        def serialize_section(section: TemplateSection) -> Dict:
            return {
                "id": section.id,
                "title": section.title,
                "template_content": section.template_content,
                "variables": section.variables,
                "required": section.required,
                "order": section.order,
                "conditions": section.conditions,
                "subsections": [serialize_section(sub) for sub in section.subsections],
                "generation_instructions": section.generation_instructions,
            }

        return {
            "document_type": template_structure["document_type"],
            "version": template_structure["version"],
            "sections": [serialize_section(s) for s in template_structure["sections"]],
        }

    def _create_input_fields(self, variable_system: Dict) -> List[Dict]:
        """Create input field definitions for UI"""

        fields = []

        for var_name, var_def in variable_system["variables"].items():
            field = {
                "name": var_name,
                "type": var_def["type"],
                "label": var_def["ui_hints"]["label"],
                "placeholder": var_def["ui_hints"]["placeholder"],
                "required": var_def["required"],
                "help_text": var_def["ui_hints"]["help_text"],
                "validation": var_def["validation"],
            }

            if var_def["type"] == "choice":
                field["options"] = var_def["ui_hints"].get("options", [])
            elif var_def["type"] == "date":
                field["date_picker"] = True

            fields.append(field)

        ordered_names = variable_system.get("input_order", [])
        fields.sort(key=lambda f: ordered_names.index(f["name"]) if f["name"] in ordered_names else 999)

        return fields

    def _create_conditional_fields(self, conditional_logic: Dict) -> List[Dict]:
        """Create conditional field definitions"""

        conditional_fields = []

        for condition_id, condition in conditional_logic["conditions"].items():
            field = {
                "condition_id": condition_id,
                "type": condition["type"],
                "expression": condition["expression"],
                "affects": self._find_affected_sections(condition_id, conditional_logic),
            }
            conditional_fields.append(field)

        return conditional_fields

    def _find_affected_sections(self, condition_id: str, conditional_logic: Dict) -> List[str]:
        """Find sections affected by a condition"""

        affected = []

        for rule in conditional_logic["rules"]:
            if rule["condition_id"] == condition_id:
                action = conditional_logic["actions"].get(rule["action_id"], {})
                if action.get("target"):
                    affected.append(action["target"])

        return affected

    def _create_section_toggles(self, template_structure: Dict) -> List[Dict]:
        """Create section toggle controls for optional sections"""

        toggles = []

        for section in template_structure["sections"]:
            if not section.required:
                toggle = {
                    "section_id": section.id,
                    "section_title": section.title,
                    "default_state": "hidden",
                    "conditions": section.conditions,
                }
                toggles.append(toggle)

        return toggles

    def _check_if_llm_required(self, template_structure: Dict) -> bool:
        """Check if LLM is required for generation"""

        for section in template_structure["sections"]:
            if "[[GENERATE:" in section.template_content:
                return True
            for subsection in section.subsections:
                if "[[GENERATE:" in subsection.template_content:
                    return True

        return False

    def _humanize_variable_name(self, var_name: str) -> str:
        """Convert variable name to human-readable label"""

        replacements = {
            "_name": " Name",
            "_date": " Date",
            "_amount": " Amount",
            "plaintiff": "Plaintiff",
            "defendant": "Defendant",
            "_": " ",
        }

        label = var_name
        for old, new in replacements.items():
            label = label.replace(old, new)

        return label.title()

    def _determine_input_type(self, var_type: str) -> str:
        """Determine HTML input type from variable type"""

        type_mapping = {
            "text": "text",
            "date": "date",
            "money": "number",
            "number": "number",
            "choice": "select",
            "boolean": "checkbox",
            "list": "textarea",
        }

        return type_mapping.get(var_type, "text")

    def _initialize_llm(self, temperature: float = 0):
        """Initialize LLM client using the shared LLMClient class"""
        return LLMClient(temperature=temperature)
