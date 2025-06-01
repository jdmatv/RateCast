import yaml
import os
from jinja2 import Template, meta
from libs.utils import logger

class PromptManager:
    def __init__(self, prompts_dir="prompts"):
        self.prompts_dir = prompts_dir
        self.prompts = {}
        self._load_all_prompts()

    def _load_all_prompts(self):
        """Loads all YAML prompt files from the specified directory."""
        if not os.path.isdir(self.prompts_dir):
            raise FileNotFoundError(f"Prompt directory not found: {self.prompts_dir}")

        for filename in os.listdir(self.prompts_dir):
            if filename.endswith(('.yaml', '.yml')):
                filepath = os.path.join(self.prompts_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        file_prompts = yaml.safe_load(f)
                        self.prompts.update(file_prompts)
                    except yaml.YAMLError as e:
                        logger.error(f"Error loading YAML from {filename}: {e}")
                        continue
    
    def _get_template_variables(self, template_string: str) -> set:
        """
        Extracts the names of variables required by a Jinja2 template string.
        """
        env = Template(template_string).environment
        parsed_content = env.parse(template_string)
        return meta.find_undeclared_variables(parsed_content)

    def get_prompt(self, prompt_name: str):
        """
        Retrieves a prompt template by its name.
        Returns a dictionary with 'system_message' and 'user_template' (if applicable).
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found.")
        return self.prompts[prompt_name]

    def render_prompt(self, prompt_name: str, **kwargs) -> list[dict]:
        """
        Renders the system and user message templates for a given prompt,
        inserting provided variables.
        Returns a list of message dictionaries suitable for chat-based LLMs.
        """
        prompt_data = self.get_prompt(prompt_name)
        messages = []

        all_expected_fields = set()
        rendered_field_info = []

        if 'system_message' in prompt_data and prompt_data['system_message']:
            system_template_string = prompt_data['system_message']
            system_template = Template(system_template_string)
            system_expected_fields = self._get_template_variables(system_template_string)
            all_expected_fields.update(system_expected_fields)

            # Check for missing system fields
            missing_system_fields = system_expected_fields - kwargs.keys()
            if missing_system_fields:
                logger.warning(f"Prompt '{prompt_name}': Missing system template arguments: {', '.join(missing_system_fields)}")
            
            rendered_system_content = system_template.render(**kwargs)
            messages.append({
                "role": "system",
                "content": rendered_system_content
            })

            rendered_field_info.append(f"System Message (length: {len(rendered_system_content.split())} words)")

        if 'user_template' in prompt_data and prompt_data['user_template']:
            user_template_string = prompt_data['user_template']
            user_template = Template(user_template_string)
            user_expected_fields = self._get_template_variables(user_template_string)
            all_expected_fields.update(user_expected_fields)

            # Check for missing user fields
            missing_user_fields = user_expected_fields - kwargs.keys()
            if missing_user_fields:
                logger.warning(f"Prompt '{prompt_name}': Missing user template arguments: {', '.join(missing_user_fields)}")

            rendered_user_content = user_template.render(**kwargs)
            messages.append({
                "role": "user",
                "content": rendered_user_content
            })
            rendered_field_info.append(f"User Message (length: {len(rendered_user_content)} chars)")

        # Check for unused arguments (too many args provided)
        provided_args = set(kwargs.keys())
        unused_args = provided_args - all_expected_fields
        if unused_args:
            logger.warning(f"Prompt '{prompt_name}': Too many arguments provided. Unused: {', '.join(unused_args)}")

        # Log summary of fields found and their lengths
        total_fields_expected = len(all_expected_fields)
        total_fields_found_in_kwargs = len(provided_args.intersection(all_expected_fields))

        logger.info(
            f"Prompt '{prompt_name}': Found {total_fields_found_in_kwargs} of {total_fields_expected} expected fields. "
            f"Rendered content: {'; '.join(rendered_field_info)}"
        )
        
        return messages

# Initialize the prompt manager once
prompt_manager = PromptManager()