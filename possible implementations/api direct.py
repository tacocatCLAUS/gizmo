import requests
import json


apiurl = "http://localhost:11434/api/chat"  # your API URL

payload = {
    "model": "mistral:7b",
    "system": "You are a helpful assistant named Gizmo.",
    "messages": [
        {"role": "user", "content": "what is the capital of france"},
        {"role": "assistant", "content": "paris"},
        {"role": "user", "content": "what about germany"}
    ]
}

response = requests.post(apiurl, json=payload)

if response.ok:
    try:
        # Attempt to parse the response as JSONL
        messages = [json.loads(line) for line in response.text.splitlines()]
        print("Response from API:")
        for message in messages:
            print(message)
    except json.JSONDecodeError:
        print("Failed to decode JSONL response.")
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)
