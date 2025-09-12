import os, httpx

API_KEY = os.getenv("API_TOKEN", os.getenv("TEST_API_KEY", "test"))
BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")

async def main():
    headers = {"X-API-Key": API_KEY}
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as ac:
        # status first
        r = await ac.get("/llm/status", headers=headers)
        print("status:", r.status_code, r.text)
        if r.json().get("configured") is not True:
            print("LLM not configured (set GROQ_API_KEY)")
            return
        payload = {
            "messages": [
                {"role": "user", "content": "Hello, give me one word: ping"}
            ]
        }
        r2 = await ac.post("/llm/chat", headers=headers, json=payload)
        print("chat:", r2.status_code)
        print(r2.text)

if __name__ == "__main__":
    import anyio
    anyio.run(main)
