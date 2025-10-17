""" 
Output: 

N = 5
Total time taken with threads: 6.49702787399292 seconds
Total time taken without threads: 25.99307918548584 seconds

N = 20
Total time taken with threads: 11.6678946018219 seconds
Total time taken without threads: 114.68130874633789 seconds
"""

import threading 
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

def run_model(prompt: str) -> str:
    client = OpenAI(api_key = OPENAI_KEY)
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    return response.output_text

def prompt(name: str):
    return f"Tell me a joke about {name}."

def task(name: str):
    response = run_model(prompt(name))
    print(f"Response for {name}: {response}")

names = ["Alice", "Bob", "Charlie", "David", "Eve"]
names_20 = names * 4  

if __name__ == "__main__":
    start_threads = time.time()
    threads = []
    for name in names:
        thread = threading.Thread(target=task, args=(name,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    end_threads = time.time()

    start_no_threads = time.time()
    for name in names:
        response = run_model(name)
        print(f"Response for {name}: {response}")
    end_no_threads = time.time()

    print(f"Total time taken with threads: {end_threads - start_threads} seconds")
    print(f"Total time taken without threads: {end_no_threads - start_no_threads} seconds")