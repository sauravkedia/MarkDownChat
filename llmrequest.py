import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "llama3",
    "messages": [
        {
            "role": "system",
            "content": "You are an insurance assistant."
        },
        {
            "role": "user",
            "content": "What is term insurance?"
        }
    ],
    "stream": False
}

response = requests.post(url, json=payload)

print(response.json()["message"]["content"])


class OllamaClient:
    @staticmethod
    def chat(messages, model="llama3", stream=False):
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        response = requests.post(url, json=payload)
        return response.json()