import requests

prompt = "Explain Azerbaijan Armenia conflict in 2 sentences."

res = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3:8b",
    "prompt": prompt,
    "stream": False
})

print(res.json()["response"])
