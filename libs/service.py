import os
import json
import csv
import ollama
from datetime import datetime
import requests
from typing import Optional


LOGS_DIR = "logs"
LLM_USAGE_CSV_FILE = os.path.join(LOGS_DIR, "llm_usage.csv")
CURRENT_BALANCES_JSON_FILE = os.path.join(LOGS_DIR, "current_token_balances.json")

INITIAL_TOKEN_BALANCES = {
    "SONNET_3_7_BALANCE": float(os.getenv("SONNET37_BALANCE")),
    "SONNET_4_BALANCE": float(os.getenv("SONNET4_BALANCE")),
    "O4_MINI_BALANCE": float(os.getenv("OPENAI_04MINI_BALANCE")),
    "O3_BALANCE": float(os.getenv("OPENAI_03_BALANCE")),
    "GPT_4_1_MINI_BALANCE": float(os.getenv("OPENAI_41MINI_BALANCE")),
}

MODEL_NAME_DIC = {
    'sonnet 4': 'claude-sonnet-4-20250514',
    'sonnet 3.7': 'claude-3-7-sonnet-latest',
    'o4-mini': 'o4-mini',
    'o3': 'o3',
    'gpt-4.1-mini': 'gpt-4.1-mini',
    'qwen3:0.6b': 'ollama/qwen3:0.6b',
    'qwen3:1.7b': 'ollama/qwen3:1.7b',
    'qwen3:4b': 'ollama/qwen3:4b',
    'qwen3:8b': 'ollama/qwen3:8b',
    'qwen3:14b': 'ollama/qwen3:14b',
    'llama3.2': 'ollama/llama3.2:latest'
}

def map_api_model_to_balance_key(api_model_name, initial_balance_keys):
    """
    Tries to map an API model name (e.g., 'claude-3-5-sonnet-20240620')
    to a key in INITIAL_TOKEN_BALANCES (e.g., 'sonnet 3.7', 'o4-mini').
    This is heuristic and might need refinement based on actual keys and model names.
    """
    api_model_lower = api_model_name.lower()
    
    # Prioritize direct or very close matches
    for key in initial_balance_keys:
        if api_model_lower == key.lower():
            return key
        if api_model_lower.replace("-", "") == key.lower().replace("-", "").replace(" ", ""):
            return key

    # Heuristic matching
    if "claude-3-7-sonnet" in api_model_lower:
        for key in initial_balance_keys:
            if "sonnet" in key.lower() and "3_7" in key: return key
    elif "claude-sonnet-4" in api_model_lower:
        for key in initial_balance_keys:
            if "sonnet" in key.lower() and "4" in key: return key
            
    elif "gpt-4.1-mini" in api_model_lower:
        for key in initial_balance_keys:
            if "gpt_4_1_mini" in key.lower(): return key
    elif "o3" in api_model_lower and "o3_mini" not in api_model_lower:
        for key in initial_balance_keys:
            if "o3" in key.lower() and "o3_mini" not in api_model_lower: return key
    elif "o4-mini" in api_model_lower: 
        for key in initial_balance_keys:
            if "o4_mini" in key.lower(): return key

    # Fallback: if no specific mapping, return None or a default key.
    # This means balance tracking might not work for this model if no key is found.
    print(f"Warning: Could not map API model '{api_model_name}' to a known balance key.")
    return None

class CompletionsService:
    def __init__(self):
        self.metuculus_token = os.getenv('METACULUS_TOKEN')
        self.initial_token_balances = INITIAL_TOKEN_BALANCES 
        self.current_token_balances = self._load_or_initialize_current_balances()
        self.openai_base_url = "https://llm-proxy.metaculus.com/proxy/openai/v1/chat/completions/"
        self.anthropic_base_url = "https://llm-proxy.metaculus.com/proxy/anthropic/v1/messages/"
        
        # Ensure logs directory exists
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        
        # Initialize CSV log file if it doesn't exist
        if not os.path.isfile(LLM_USAGE_CSV_FILE):
            with open(LLM_USAGE_CSV_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "model", "provider", "input_tokens", "output_tokens", "remaining_tokens_for_key", "balance_key_used"])

    def _load_or_initialize_current_balances(self):
        """Loads current balances from JSON file or initializes from INITIAL_TOKEN_BALANCES."""
        if os.path.exists(CURRENT_BALANCES_JSON_FILE):
            try:
                with open(CURRENT_BALANCES_JSON_FILE, 'r') as f:
                    loaded_balances = json.load(f)
                    for key, initial_value in self.initial_token_balances.items():
                        if key not in loaded_balances:
                            loaded_balances[key] = float(initial_value) 
                    return loaded_balances
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading balances from {CURRENT_BALANCES_JSON_FILE}: {e}. Re-initializing.")
        
        return {key: float(value) for key, value in self.initial_token_balances.items()}

    def _save_current_balances(self):
        """Saves the current token balances to the JSON file."""
        try:
            with open(CURRENT_BALANCES_JSON_FILE, 'w') as f:
                json.dump(self.current_token_balances, f, indent=4)
        except IOError as e:
            print(f"Error saving balances to {CURRENT_BALANCES_JSON_FILE}: {e}")

    def _update_balances_and_log(self, api_model_name, provider, input_tokens, output_tokens):
        """Updates balances and logs the usage to the CSV file."""
        balance_key_used = map_api_model_to_balance_key(api_model_name, list(self.initial_token_balances.keys()))
        remaining_balance_for_key = "N/A"
        
        if balance_key_used and balance_key_used in self.current_token_balances:
            tokens_consumed = (input_tokens or 0) + (output_tokens or 0)
            current_balance_val = float(self.current_token_balances.get(balance_key_used, 0.0))
            current_balance_val -= tokens_consumed
            self.current_token_balances[balance_key_used] = current_balance_val
            remaining_balance_for_key = current_balance_val
            self._save_current_balances()
        elif balance_key_used:
            print(f"Warning: Balance key '{balance_key_used}' (mapped from '{api_model_name}') not found in current balances. Initializing from defaults if possible.")
            if balance_key_used in self.initial_token_balances:
                initial_val = float(self.initial_token_balances[balance_key_used])
                tokens_consumed = (input_tokens or 0) + (output_tokens or 0)
                new_balance = initial_val - tokens_consumed
                self.current_token_balances[balance_key_used] = new_balance
                remaining_balance_for_key = new_balance
                self._save_current_balances()
            else:
                print(f"Error: Balance key '{balance_key_used}' also not in initial_token_balances. Cannot track balance.")
        else:
            print(f"Warning: No balance key found for model '{api_model_name}'. Balance tracking skipped for this call.")

        timestamp = datetime.now().isoformat()
        try:
            with open(LLM_USAGE_CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, api_model_name, provider, input_tokens or 0, output_tokens or 0, remaining_balance_for_key, balance_key_used or "N/A"])
        except IOError as e:
            print(f"Error writing to CSV log {LLM_USAGE_CSV_FILE}: {e}")
        return remaining_balance_for_key

    def _make_request(self, url, headers, data):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            # Try to parse error details if JSON
            try:
                error_details = e.response.json()
                print(f"Error details: {error_details}")
            except json.JSONDecodeError:
                pass # No JSON in error response body
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response: {e}")
            print(f"Response text: {response.text if 'response' in locals() else 'No response object'}")
            raise

    def get_openai_completion(self, model_name, messages, temperature=None, max_tokens=None):
        headers = {
            "Authorization": f"Token {self.metuculus_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": messages
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response_data = self._make_request(self.openai_base_url, headers, payload)
            
            completion_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            
            self._update_balances_and_log(model_name, "OpenAI", input_tokens, output_tokens)
            
            return completion_content
        except Exception as e:
            print(f"Error in get_openai_completion: {e}")
            # Log a failed attempt if possible, though token counts might be unknown
            self._update_balances_and_log(model_name, "OpenAI_Error", 0, 0) # Or some other way to denote failure
            raise

    def get_anthropic_completion(self, model_name, messages, temperature=None, max_tokens=None):
        headers = {
            "Authorization": f"Token {self.metuculus_token}",
            "anthropic-version": "2023-06-01", # As per your curl example
            "Content-Type": "application/json"
        }

        # Handle system prompt for Anthropic
        # Anthropic API expects 'system' prompt as a top-level parameter,
        # not within the 'messages' list like OpenAI.
        system_prompt = None
        processed_messages = []
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content")
            processed_messages = messages[1:]
        else:
            processed_messages = messages

        payload = {
            "model": model_name,
            "messages": processed_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature # Anthropic also uses 'temperature'
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens # Anthropic v1/messages API uses 'max_tokens'
        else:
            payload["max_tokens"] = 20000

        try:
            response_data = self._make_request(self.anthropic_base_url, headers, payload)
            
            # Anthropic response structure for content:
            # response_data['content'] is a list of content blocks.
            # We'll concatenate text blocks.
            completion_content = ""
            if response_data.get("content"):
                for block in response_data["content"]:
                    if block.get("type") == "text":
                        completion_content += block.get("text", "")
            
            usage = response_data.get("usage", {})
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            
            self._update_balances_and_log(model_name, "Anthropic", input_tokens, output_tokens)
            
            return completion_content
        except Exception as e:
            print(f"Error in get_anthropic_completion: {e}")
            self._update_balances_and_log(model_name, "Anthropic_Error", 0, 0)
            raise
    
    def get_ollama_completion(self, model_name, messages):
        """
        Get completion from Ollama model.
        """
        try:
            model_name = model_name.split('ollama/')[-1]
            response = ollama.chat(model=model_name, messages=messages)
            completion_content = response['message']['content']
            clean_output = completion_content.split('</think>')[-1].strip()
            return clean_output
        except Exception as e:
            print(f"Error in get_ollama_completion: {e}")
            self._update_balances_and_log(model_name, "Ollama_Error", 0, 0)

    def get_completion(
            self, 
            model_name: str, 
            messages: list[dict], 
            temperature: Optional[float]=None, 
            max_tokens: Optional[int]=None
    ) -> str:
        """
        Generic method to get completion. Determines provider based on model name.
        """
        full_model_name = MODEL_NAME_DIC.get(model_name, model_name)

        if "claude" in full_model_name.lower():
            return self.get_anthropic_completion(full_model_name, messages, temperature, max_tokens)
        elif any(keyword in full_model_name.lower() for keyword in ["gpt", "o3", "o4"]):
            return self.get_openai_completion(full_model_name, messages, temperature, max_tokens)
        elif "ollama" in full_model_name.lower():
            return self.get_ollama_completion(full_model_name, messages)
        else:
            print(f"Warning: Provider for model '{full_model_name}' not explicitly determined. Defaulting to OpenAI.")
            return self.get_openai_completion(full_model_name, messages, temperature, max_tokens)