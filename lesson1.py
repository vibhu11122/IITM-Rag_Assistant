from sentence_transformers import SentenceTransformer, util

#very small model that just runs on the laptop 
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences= [
    "Students must have 75% attendance to write the end-term exam.",
    "Can I skip classes and still sit for the final?",
    "The fee is refunded only if you withdraw before the deadline.",
    "I love playing badminton on weekends.",
]

embeddings =  model.encode(sentences)

print("shape:", embeddings.shape)
print("First 10 numbers:", embeddings[0][:10])

scores = util.cos_sim(embeddings, embeddings)
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"{scores[i][j]:.2f}  |  {sentences[i][:40]}  <->  {sentences[j][:40]}")