import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role":"system","content":"You are a cautious planner."},
        {"role":"user","content":"Propose a small refactor within policy limits."}
    ],
    temperature=0
)
print(resp.choices[0].message.content)
