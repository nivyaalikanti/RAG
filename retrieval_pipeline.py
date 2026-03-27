from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Config
PERSIST_DIRECTORY = "db/chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# 2. Load Vector Store
def load_vector_store():
    print("Loading vector store...")

    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Vector store loaded")
    return db


# 3. Create Retriever
def get_retriever(db, k=5):
    return db.as_retriever(search_kwargs={"k": k})


# 4. Query + Retrieve
def retrieve(query, retriever):
    print(f"\nUser Query: {query}")
    docs = retriever.invoke(query)

    print("\n--- Context ---")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(doc.page_content)

    return docs


# 5. Main
def main():
    db = load_vector_store()
    retriever = get_retriever(db)

    query = "How much did Microsoft pay to acquire GitHub?"
    retrieve(query, retriever)


if __name__ == "__main__":
    main()