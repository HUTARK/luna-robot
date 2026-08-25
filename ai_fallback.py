import json, os

AI_CONFIG = "/home/groot/pi_assistant/ai_config.json"

def get_api_key():
    if os.path.exists(AI_CONFIG):
        try:
            with open(AI_CONFIG, 'r') as f:
                return json.load(f).get("openai_key", "")
        except: pass
    return ""

def ask_gpt(question: str, language: str = "en") -> str:
    key = get_api_key()
    if not key:
        return "My online brain is not configured yet."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        system = (
            "You are Ben, a friendly robot assistant. "
            "Answer in one or two short sentences, simple and spoken-friendly. "
            "No markdown, no lists."
        )
        if language == "ta":
            system += " Reply in Tamil."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question}
            ],
            max_tokens=100,
            timeout=15
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT] Error: {e}")
        return "Sorry, I could not reach my online brain right now."
