import yaml
import os
from jinja2 import Template

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
                        print(f"Loaded prompts from: {filename}")
                    except yaml.YAMLError as e:
                        print(f"Error loading YAML from {filename}: {e}")
                        continue

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

        if 'system_message' in prompt_data and prompt_data['system_message']:
            system_template = Template(prompt_data['system_message'])
            messages.append({
                "role": "system",
                "content": system_template.render(**kwargs)
            })

        if 'user_template' in prompt_data and prompt_data['user_template']:
            user_template = Template(prompt_data['user_template'])
            messages.append({
                "role": "user",
                "content": user_template.render(**kwargs)
            })

        return messages

# Initialize the prompt manager once
prompt_manager = PromptManager()