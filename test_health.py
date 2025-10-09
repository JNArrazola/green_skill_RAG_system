import numpy as np
import pandas as pd

skills_df = pd.read_csv("data/jan_to_apr_2025_with_languages_cleaned.csv")  # o el dataset original
embeddings = np.load("data/embeddings/job_skills_embeddings.npy")

# ids originales
original_ids = set(skills_df.index.tolist())

# ids presentes en el .npy (suponiendo que usaste el mismo orden)
present_ids = set(range(len(embeddings)))

missing = sorted(list(original_ids - present_ids))
print("Faltan:", missing)