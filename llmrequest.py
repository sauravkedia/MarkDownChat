import requests

# =========================================================
# OLLAMA REST API CLIENT
# =========================================================

class OllamaRestLLM:

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float = 0
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def invoke(self, prompt: str):

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }


        # print(f"payload: {payload}")

        response = requests.post(url, json=payload, timeout=300)

        response.raise_for_status()

        result = response.json()

        return result["response"]