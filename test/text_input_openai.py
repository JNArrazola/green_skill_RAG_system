from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")

client = OpenAI(api_key = OPENAI_KEY)

response = client.responses.create(
        model="gpt-5-mini",
        input="Hola, me llamo Joshua, ¿cómo me llamo?"
)

print(response.output_text)