import openai
import os

class LLMService:
    def __init__(self, model):
        self.client = openai.OpenAI(api_key=os.getenv("METACULUS_TOKEN"),
                                    base_url="https://llm-proxy.metaculus.com/proxy/openai/v1/chat/completions/")
        self.model = model

    def generate_chat_response(self, messages, temperature=0.7, max_tokens=500):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during LLM call: {e}")
            raise


# test the service
