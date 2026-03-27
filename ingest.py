import chromadb
from embedder import get_embedding
from rag_data import docs

chroma = chromadb.PersistentClient(path="vectordb")

collection = chroma.get_or_create_collection("docs")

for i, d in enumerate(docs):
    collection.add(
        documents=[d],
        embeddings=[get_embedding(d)],
        ids=[str(i)]
    )
