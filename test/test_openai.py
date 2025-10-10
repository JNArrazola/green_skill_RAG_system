from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")

client = OpenAI(api_key= OPENAI_KEY)

response = client.embeddings.create(
    input="Joshua",
    model ="text-embedding-3-large"
)

print(response.data[0])