import requests
from typing import Optional, Dict, Any

class OllamaClient:
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
    def __call__(self, query: str, sp: Optional[str] = None) -> str:
        # Combine system prompt and query if provided
        prompt = f"{sp}\n{query}" if sp else query
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]

class OllamaChat:
    def __init__(self, model: str = "mistral", sp: Optional[str] = None, base_url: str = "http://localhost:11434"):
        self.model = model
        self.sp = sp
        self.base_url = base_url
        self.history = []
        
    def __call__(self, query: str) -> str:
        messages = self.history + [{"role": "user", "content": query}]
        formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        if self.sp:
            formatted_prompt = f"{self.sp}\n{formatted_prompt}"
            
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": formatted_prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        response_text = response.json()["response"]
        
        self.history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response_text}
        ])
        return response_text

    def toolloop(self, query: str) -> str:
        return self(query)
