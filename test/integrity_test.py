from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import os

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=OPENAI_KEY)

def generate_embedding(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)

def verify(a: np.ndarray, original_data: list[str], positions: list[int]) -> bool:
    all_ok = True
    for pos in positions:
        emb = generate_embedding(original_data[pos])
        diff = np.abs(a[pos] - emb)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        if np.allclose(a[pos], emb, atol=1e-6):
            print(f"Position {pos} OK (max diff={max_diff:.2e}, mean diff={mean_diff:.2e})")
        else:
            print(f"Mismatch at position {pos} (max diff={max_diff:.2e}, mean diff={mean_diff:.2e})")
            all_ok = False
    return all_ok

npy_array_1 = np.load("../data/embeddings/job_skills_embeddings_first_part_partial.npy")
original_data = pd.read_csv("../data/full_dataset/jul24_to_jul_2025_cleaned.csv")["Skills"].tolist()

print("Shape inicial:", npy_array_1.shape)
verify(npy_array_1, original_data, [0, 100, 5000, 25000, 51000])

missing_index = 51001
new_embedding = generate_embedding(original_data[missing_index]).reshape(1, -1)
npy_array_1 = np.concatenate([npy_array_1, new_embedding], axis=0)

print("final shape: ", npy_array_1.shape)

verify(npy_array_1, original_data, [0, 100, 5000, 25000, 51000, missing_index])

np.save("../data/embeddings/job_skills_embeddings_first_part_fixed_51002.npy", npy_array_1)
print("Saved")
