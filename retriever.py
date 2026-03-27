import chromadb
from embedder import get_embedding

chroma = chromadb.PersistentClient(path="vectordb")
collection = chroma.get_collection("docs")

def retrieve(query, k=3):
    result = collection.query(
        query_embeddings=[get_embedding(query)],
        n_results=k
    )
    return result["documents"][0]

print('DONE!')
