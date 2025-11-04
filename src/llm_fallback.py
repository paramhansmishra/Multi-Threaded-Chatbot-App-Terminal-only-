"""
LLM Fallback Module
-------------------
Tries OpenRouter (free-tier proxy to models like DeepSeek, Gemini, Mistral).
If no API key or no internet, uses a local text-generation stub so the bot
keeps talking gracefully.
"""
#Key: bash - setx OPENROUTER_API_KEY"sk-or-v1-xxxxx....." --> make your own key. I wont gib you mine >_<
import os, requests

# optional: set OPENROUTER_API_KEY as an environment variable once
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

def get_llm_response(prompt: str) -> str:
    """
    Return an LLM-generated answer for a given prompt.
    Falls back to a local stub if the network call fails.
    """
    # if you have a key, try the live model
    if OPENROUTER_KEY:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "mistralai/mistral-7b-instruct",  # free-tier model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        try:
            resp = requests.post(BASE_URL, headers=headers, json=data, timeout=10)
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"].strip()
            return reply
        except Exception as e:
            print(f"⚠️  Fallback to offline mode ({e})")

    # offline (simple heuristic)
    return (
        "I'm not sure about that. It might be outside my stored knowledge, "
        "but if you give me a hint I can reason it out with you."
    )
