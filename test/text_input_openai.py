from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")

client = OpenAI(api_key = OPENAI_KEY)



response = client.responses.create(
  model="gpt-4.1",
  input="Si me llamo Joshua, y me subo a un tren, ¿cómo me llamo?"
)

print(response.output[0].content[0].text)