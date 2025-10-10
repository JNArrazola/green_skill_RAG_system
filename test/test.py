from sentence_transformers import SentenceTransformer, util
sentences = ["America", "United States"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
print(util.cos_sim(embeddings[0], embeddings[1]))