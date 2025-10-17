import faiss 

def load_index(index_path):
    index = faiss.read_index(index_path)
    return index

INDEX_PATH_1 = "../data/embeddings/altLabel_skill_embeddings.index"
INDEX_PATH_2 = "../data/embeddings/esco_green_skills_text-embedding-3-large.index"

index1 = load_index(INDEX_PATH_1)
index2 = load_index(INDEX_PATH_2)

print("Index 1 loaded with", index1.ntotal, "vectors.")
print("Index 2 loaded with", index2.ntotal, "vectors.")
