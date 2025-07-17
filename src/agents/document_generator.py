import json
import re
from datetime import datetime
from typing import Any, Dict, List, Set

from src.clients import LLMClient
from src.logger import logger


class DocumentGenerator:
    def __init__(self):
        self.llm = self._initialize_llm()

    def _initialize_llm(self, temperature: float = 0):
        return LLMClient(temperature=temperature)

    def generate_document(self, template: Dict, variable_values: Dict[str, Any]) -> str:
        """
        Generate a new document from template and variable values.

        Args:
            template: The executable template dictionary
            variable_values: Dictionary mapping variable names to their values

        Returns:
            Generated document as formatted text
        """

        self._validate_inputs(template, variable_values)

        processed_values = self._process_variables(template, variable_values)

        active_sections = self._evaluate_conditionals(template, processed_values)

        document_parts = []

        if template.get("metadata", {}).get("document_type"):
            document_parts.append(self._generate_document_header(template, processed_values))

        structure = template.get("structure", {})
        for section in structure.get("sections", []):
            if self._should_include_section(section, active_sections):
                section_content = self._generate_section(section, processed_values, template.get("style_guide", {}))
                if section_content:
                    document_parts.append(section_content)

        document = self._assemble_document(document_parts, template.get("style_guide", {}))
        document = self._post_process_document(document, template, processed_values)

        return document

    def _validate_inputs(self, template: Dict, variable_values: Dict[str, Any]):
        """Validate that all required variables are provided and valid"""

        variable_system = template.get("variable_system", {})
        variables = variable_system.get("variables", {})

        for var_name, var_def in variables.items():
            if var_def.get("required", True) and var_name not in variable_values:
                raise ValueError(f"Required variable '{var_name}' ({var_def['ui_hints']['label']}) is missing")

        validation_rules = template.get("validation_rules", {}).get("variable_validation", {})

        for var_name, value in variable_values.items():
            if var_name in variables:
                var_def = variables[var_name]

                self._validate_variable_type(var_name, value, var_def)

                if var_name in validation_rules:
                    self._apply_validation_rules(var_name, value, validation_rules[var_name])

        self._validate_relationships(template, variable_values)

    def _validate_variable_type(self, var_name: str, value: Any, var_def: Dict):
        """Validate variable type"""

        expected_type = var_def["type"]

        if expected_type == "text" and not isinstance(value, str):
            raise ValueError(f"Variable '{var_name}' must be text, got {type(value)}")

        elif expected_type == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"Variable '{var_name}' must be a number")

        elif expected_type == "date":
            if not isinstance(value, (str, datetime)):
                raise ValueError(f"Variable '{var_name}' must be a date")

        elif expected_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Variable '{var_name}' must be true or false")

    def _apply_validation_rules(self, var_name: str, value: Any, rules: Dict):
        """Apply custom validation rules to a variable"""

        if "min" in rules and isinstance(value, (int, float)) and value < rules["min"]:
            raise ValueError(f"Variable '{var_name}' must be at least {rules['min']}")

        if "max" in rules and isinstance(value, (int, float)) and value > rules["max"]:
            raise ValueError(f"Variable '{var_name}' must be at most {rules['max']}")

        if "pattern" in rules and isinstance(value, str):
            if not re.match(rules["pattern"], value):
                raise ValueError(f"Variable '{var_name}' does not match required pattern: {rules['pattern']}")

        if "max_length" in rules and isinstance(value, str):
            if len(value) > rules["max_length"]:
                raise ValueError(f"Variable '{var_name}' exceeds maximum length of {rules['max_length']}: {value}")

    def _validate_relationships(self, template: Dict, variables: Dict[str, Any]):
        """Validate relationships between variables"""

        relationships = template.get("variable_system", {}).get("relationships", [])

        for rel in relationships:
            if rel["type"] == "temporal":
                before_var = rel.get("before")
                after_var = rel.get("after")

                if before_var in variables and after_var in variables:
                    before_date = self._parse_date(variables[before_var])
                    after_date = self._parse_date(variables[after_var])

                    if after_date < before_date:
                        raise ValueError(f"{after_var} must be after {before_var}")

            elif rel["type"] == "logical":
                if "condition" in rel:
                    if not self._evaluate_condition_expression(rel["condition"], variables):
                        raise ValueError(f"Logical constraint violated: {rel.get('rule', 'Unknown rule')}")

    def _parse_date(self, date_value: Any):
        """Parse date from various formats"""
        from datetime import datetime

        if isinstance(date_value, datetime):
            return date_value

        if isinstance(date_value, str):
            for fmt in ["%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%B %d, %Y"]:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue

        return str(date_value)

    def _process_variables(self, template: Dict, variable_values: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format variables according to template rules"""

        processed = {}
        variable_system = template.get("variable_system", {})
        variables = variable_system.get("variables", {})
        style_guide = template.get("style_guide", {})

        for var_name, value in variable_values.items():
            if var_name in variables:
                var_def = variables[var_name]

                if var_def["type"] == "date":
                    processed[var_name] = self._format_date(value, style_guide.get("formatting", {}))
                elif var_def["type"] == "money":
                    processed[var_name] = self._format_money(value, style_guide.get("formatting", {}))
                elif var_def["type"] == "text":
                    processed[var_name] = self._format_text(value, var_def)
                else:
                    processed[var_name] = value
            else:
                processed[var_name] = value

        processed.update(self._generate_derived_variables(template, processed))

        return processed

    def _format_date(self, date_value: Any, formatting: Dict) -> str:
        """Format date according to style guide"""

        date_format = formatting.get("date_format", "DD.MM.YYYY")
        date_obj = self._parse_date(date_value)

        if isinstance(date_obj, datetime):
            if date_format == "DD.MM.YYYY":
                return date_obj.strftime("%d.%m.%Y")
            elif date_format == "YYYY-MM-DD":
                return date_obj.strftime("%Y-%m-%d")
            elif date_format == "Month DD, YYYY":
                return date_obj.strftime("%B %d, %Y")

        return str(date_value)

    def _format_money(self, amount: Any, formatting: Dict) -> str:
        """Format monetary amount according to style guide"""

        currency_format = formatting.get("currency_format", "€X,XXX")

        try:
            amount_float = float(str(amount).replace(",", "").replace("€", "").replace("$", ""))
        except ValueError:
            return str(amount)

        # Format based on style
        if currency_format.startswith("€"):
            # European format
            formatted = f"{amount_float:,.2f}".replace(",", " ").replace(".", ",")
            return f"€{formatted}"
        elif currency_format.startswith("$"):
            return f"${amount_float:,.2f}"
        else:
            return str(amount)

    def _format_text(self, value: str, var_def: Dict) -> str:
        """Format text according to variable definition"""

        if var_def.get("transform") == "uppercase":
            return value.upper()
        elif var_def.get("transform") == "lowercase":
            return value.lower()
        elif var_def.get("transform") == "title":
            return value.title()

        return value

    def _generate_derived_variables(self, template: Dict, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Generate any derived variables"""

        derived = {}

        if "hire_date" in variables and "dismissal_date" in variables:
            hire = self._parse_date(variables["hire_date"])
            dismissal = self._parse_date(variables["dismissal_date"])

            if isinstance(hire, datetime) and isinstance(dismissal, datetime):
                duration = dismissal - hire
                years = duration.days // 365
                months = (duration.days % 365) // 30
                derived["employment_duration"] = f"{years} years and {months} months"

        return derived

    def _evaluate_conditionals(self, template: Dict, variables: Dict[str, Any]) -> Set[str]:
        """Evaluate conditional logic to determine which sections to include"""

        active_sections = set()
        conditional_logic = template.get("conditional_logic", {})

        structure = template.get("structure", {})
        for section in structure.get("sections", []):
            if section.get("required", True):
                active_sections.add(section["id"])

        for rule in conditional_logic.get("rules", []):
            condition = conditional_logic["conditions"].get(rule["condition_id"], {})

            if self._evaluate_condition(condition, variables):
                action = conditional_logic["actions"].get(rule["action_id"], {})

                if action["type"] == "include_section":
                    active_sections.add(action["target"])
                elif action["type"] == "exclude_section":
                    active_sections.discard(action["target"])

        return active_sections

    def _evaluate_condition(self, condition: Dict, variables: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""

        condition_type = condition.get("type", "custom")

        if condition_type == "always":
            return True

        elif condition_type == "if_provided":
            required_vars = condition.get("evaluation", {}).get("variables", [])
            return all(variables.get(var) for var in required_vars)

        elif condition_type == "if_true":
            var_name = condition.get("evaluation", {}).get("variables", [None])[0]
            return bool(variables.get(var_name))

        elif condition_type == "custom":
            expression = condition.get("expression", "")
            return self._evaluate_condition_expression(expression, variables)

        return False

    def _evaluate_condition_expression(self, expression: str, variables: Dict[str, Any]) -> bool:
        """Evaluate a custom condition expression"""

        if expression in variables:
            return bool(variables[expression])

        if "==" in expression:
            parts = expression.split("==")
            if len(parts) == 2:
                var_name = parts[0].strip()
                expected = parts[1].strip().strip("\"'")
                return str(variables.get(var_name, "")) == expected

        elif "!=" in expression:
            parts = expression.split("!=")
            if len(parts) == 2:
                var_name = parts[0].strip()
                expected = parts[1].strip().strip("\"'")
                return str(variables.get(var_name, "")) != expected

        elif ">" in expression:
            parts = expression.split(">")
            if len(parts) == 2:
                var_name = parts[0].strip()
                try:
                    threshold = float(parts[1].strip())
                    var_value = float(variables.get(var_name, 0))
                    return var_value > threshold
                except (ValueError, TypeError):
                    return False

        expression_lower = expression.lower()

        if "discrimination" in expression_lower:
            return any(
                [
                    variables.get("discrimination_claimed", False),
                    "discrimination" in str(variables.get("dismissal_reason", "")).lower(),
                ]
            )

        return False

    def _should_include_section(self, section: Dict, active_sections: Set[str]) -> bool:
        """Determine if a section should be included"""

        if section.get("required", True):
            return True
        return section["id"] in active_sections

    def _generate_document_header(self, template: Dict, variables: Dict[str, Any]) -> str:
        """Generate document header"""

        header_parts = []
        metadata = template.get("metadata", {})

        # Document title
        doc_type = metadata.get("document_type", "Legal Document")
        title = doc_type.upper().replace("_", " ")
        header_parts.append(f"# {title}")

        if "case_number" in variables:
            header_parts.append(f"**Case No.:** {variables['case_number']}")

        if "court" in variables:
            header_parts.append(f"**Court:** {variables['court']}")

        if "court_name" in variables:
            header_parts.append(f"**Court:** {variables['court_name']}")

        if "filing_date" in variables:
            header_parts.append(f"**Date Filed:** {variables['filing_date']}")

        if "date_filed" in variables:
            header_parts.append(f"**Date Filed:** {variables['date_filed']}")

        return "\n".join(header_parts)

    def _generate_section(self, section: Dict, variables: Dict[str, Any], style_guide: Dict) -> str:
        """Generate content for a single section"""

        content_parts = []

        title = self._format_section_title(section["title"], style_guide)
        if title:
            content_parts.append(title)

        if section.get("template_content"):
            section_content = self._fill_template(
                section["template_content"], variables, section.get("generation_instructions", {})
            )
            content_parts.append(section_content)

        for subsection in section.get("subsections", []):
            subsection_content = self._generate_section(subsection, variables, style_guide)
            if subsection_content:
                content_parts.append(subsection_content)

        def _clean_content(content: str) -> str:
            if content.startswith("{"):
                content = content[content.find(":") + 1 : -1]
            return (
                content.replace("```json", "")
                .replace("```", "")
                .replace("<", "")
                .replace(">", "")
                .replace("{", "")
                .replace("}", "")
                .replace("[", "")
                .replace("]", "")
                .replace('"', "")
            )

        section_content = ""
        for part in content_parts:
            if part is None:
                continue

            cleaned_content = _clean_content(part)
            section_content += f"\n\n{cleaned_content}"

        return section_content

    def _format_section_title(self, title: str, style_guide: Dict) -> str:
        """Format section title according to style guide"""

        formatting = style_guide.get("formatting", {})
        header_style = formatting.get("section_headers", "ALL CAPS")

        if header_style == "ALL CAPS":
            return f"## {title.upper()}"
        elif header_style == "Title Case":
            return f"## {title.title()}"
        else:
            return f"## {title}"

    def _fill_template(self, template_content: str, variables: Dict[str, Any], instructions: Dict) -> str:
        """Fill in a template with variable values"""

        filled_content = template_content

        for var_name, value in variables.items():
            patterns = [f"{{{{{var_name}}}}}", f"{{{{{var_name.upper()}}}}}", f"{{{{{var_name.lower()}}}}}"]

            for pattern in patterns:
                if pattern in filled_content:
                    if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                        try:
                            json_value = json.loads(value)
                            if isinstance(json_value, dict):
                                for key, content in json_value.items():
                                    if isinstance(content, str):
                                        extracted_value = content
                                        break
                                else:
                                    extracted_value = str(value)
                            else:
                                extracted_value = str(json_value)
                        except json.JSONDecodeError:
                            extracted_value = str(value)
                    else:
                        extracted_value = str(value)

                    filled_content = filled_content.replace(pattern, extracted_value)

        filled_content = self._process_conditionals_in_template(filled_content, variables)

        filled_content = self._process_generation_instructions(filled_content, variables, instructions)

        return filled_content

    def _process_conditionals_in_template(self, content: str, variables: Dict[str, Any]) -> str:
        """Process conditional sections in template"""

        conditional_pattern = r"<<IF\s+([^>]+)>>(.*?)<<\/IF>>"

        def evaluate_conditional(match):
            condition = match.group(1)
            conditional_content = match.group(2)

            if self._evaluate_template_condition(condition, variables):
                return conditional_content
            else:
                return ""

        content = re.sub(conditional_pattern, evaluate_conditional, content, flags=re.DOTALL)

        return content

    def _evaluate_template_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """Evaluate a condition in template"""

        if condition in variables:
            return bool(variables[condition])

        if "==" in condition:
            parts = condition.split("==")
            if len(parts) == 2:
                var_name = parts[0].strip()
                expected = parts[1].strip().strip("\"'")
                return str(variables.get(var_name, "")) == expected

        if condition.lower() == "true":
            return True
        elif condition.lower() == "false":
            return False

        return bool(variables.get(condition))

    def _process_generation_instructions(self, content: str, variables: Dict[str, Any], instructions: Dict) -> str:
        """Process generation instructions that require LLM"""

        generation_pattern = r"\[\[GENERATE:\s*([^\]]+)\]\]"

        def generate_content(match):
            instruction = match.group(1)

            if hasattr(self, "llm") and self.llm:
                prompt = f"""
                Generate legal text based on this instruction: {instruction}

                Context variables:
                {json.dumps(variables, indent=2)}

                Style requirements:
                - Formal legal language
                - Consistent with document tone
                - Precise and unambiguous

                Generate only the requested text, no explanations.
                """

                generated = self.llm.complete_with_system_prompt(
                    system_prompt="You are an AI assistant that generates legal text based on a given instruction. You are given a context and a request to generate text. You must generate the text based on the context and the request. You must generate the text in the same language as the context.",
                    user_prompt=prompt,
                )
                return generated.strip()
            else:
                return f"[Generated content for: {instruction}]"

        content = re.sub(generation_pattern, generate_content, content)

        return content

    def _assemble_document(self, parts: List[str], style_guide: Dict) -> str:
        """Assemble document parts into final document"""

        spacing = style_guide.get("formatting", {}).get("spacing", "double")

        if spacing == "double":
            separator = "\n\n"
        else:
            separator = "\n"

        document = separator.join(filter(None, parts))

        document = re.sub(r"\n\n\n+", "\n\n", document)

        return document.strip()

    def _post_process_document(self, document: str, template: Dict, variables: Dict[str, Any]) -> str:
        """Final post-processing of generated document"""

        style_guide = template.get("style_guide", {})

        if style_guide.get("formatting", {}).get("emphasis") == "**bold**":
            document = re.sub(r"\*\*([^*]+)\*\*", r"**\1**", document)

        self._validate_generated_document(document, template, variables)

        return document

    def _validate_generated_document(self, document: str, template: Dict, variables: Dict[str, Any]):
        """Validate the generated document"""

        validation_rules = template.get("validation_rules", {})

        for rule in validation_rules.get("content_validation", []):
            if rule["rule"] == "contains_required_phrases":
                for phrase in rule.get("phrases", []):
                    if phrase not in document:
                        logger.warning(f"Required phrase missing: {phrase}")

        if len(document) < 100:
            logger.warning("Generated document seems too short")

        required_sections = [
            s["title"] for s in template.get("structure", {}).get("sections", []) if s.get("required", True)
        ]

        for section_title in required_sections:
            if section_title.upper() not in document.upper():
                logger.warning(f"Required section might be missing: {section_title}")
