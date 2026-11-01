import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

A = np.array([0.9, 0.1])
B = np.array([0.8, 0.2])
C = np.array([0.1, 0.9])
print("A vs B:", round(cosine_similarity(A, B), 3))
print("A vs C:", round(cosine_similarity(A, C), 3))

#temporary dict

docs = [
    "A minimum of 75% attendance is required to appear for the end-term exam.",
    "Fees once paid are refundable only if withdrawal happens before week 2.",
    "Students must clear all foundation courses before moving to diploma level.",
    "Quiz 1 and Quiz 2 are conducted in person at designated exam centres.",
    "The final course grade is calculated from assignments, quizzes and the end-term.",
    "Students can register for a maximum of four courses per term.",
    "A student who fails a course may re-register for it in a later term.",
    "Hall tickets are released one week before the exam date.",
    "Use of unfair means in exams leads to disqualification from the course.",
    "Certificates are issued digitally after the results are declared.",
]

model=  SentenceTransformer('all-MiniLM-L6-v2')
docs_ven = model.encode(docs)

def search(query, k=3):
    q = model.encode(query)
    scores =[cosine_similarity(q, doc) for doc in docs_ven]
    top = np.argsort(scores)[::-1][:k]
    print(f"\nQ: {query}")
    for i in top:
        print(f"  {scores[i]:.3f} | {docs[i]}")

doc_norm = docs_ven / np.linalg.norm(docs_ven, axis=1, keepdims=True)

fast_search 

while True:
    query = input("\nAsk a question (or type 'quit'): ")
    if query.lower() == "quit":
        break
    search(query)


# search("Can I get my money back?")
# search("What if I fail a subject?")
# search("How many subjects can I take?")