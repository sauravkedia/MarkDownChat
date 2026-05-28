import requests
# import logging
# import http.client as http_client

# # Enable HTTPConnection debug logging
# http_client.HTTPConnection.debuglevel = 1

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # requests logging
# logging.getLogger("urllib3").setLevel(logging.DEBUG)
# logging.getLogger("urllib3").propagate = True

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

        # print(f"response: {response}")

        response.raise_for_status()

        result = response.json()

        return result["response"]