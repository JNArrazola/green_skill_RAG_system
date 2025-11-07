import pandas as pd

df_full = pd.read_csv("../data/full_dataset/jul24_to_jul_2025_cleaned_sorted.csv")
df_partial = pd.read_csv("../data/green_skills_with_GPT-4_missing.csv")

df_full_copy = df_full.copy()
df_full_copy["__original_index__"] = df_full_copy.index

df_merged = pd.merge(
    df_full_copy,
    df_partial,
    how="left",
    left_on=["Job_ID", "Skills"],
    right_on=["job_id", "job_skill"],
    indicator=True
)

df_missing = df_merged[df_merged["_merge"] == "left_only"]
df_missing = df_missing.set_index("__original_index__")
df_missing = df_missing.reindex(columns=df_full.columns)
df_missing = df_missing.sort_index()

print("Missing entries shape:", df_missing.shape)
print(df_missing.head())

df_missing.to_csv("../data/green_skills_with_GPT-4_missing.csv", index=True)
