from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
res = client.models.generate_content(
    model="gemini-pro",
    contents="Hello! Are you active?"
)

print(res.text)
