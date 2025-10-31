from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")
DEPLOYMENT = os.getenv("DEPLOYMENT")    
API_VERSION = os.getenv("API_VERSION")

# Client setup
client = OpenAI(
    base_url=f"{ENDPOINT}openai/deployments/{DEPLOYMENT}/",
    api_key=API_KEY,
    default_query={"api-version": API_VERSION},
    default_headers={"api-key": API_KEY},  
)

# Test request
response = client.chat.completions.create(
    model=DEPLOYMENT,  # deployment name
    messages=[
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Which model are we using?"},
    ],
    max_tokens=50,
)

print(response.choices[0].message.content)