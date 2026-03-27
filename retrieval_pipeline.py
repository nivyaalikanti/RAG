from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# 1. Config
PERSIST_DIRECTORY = "db/chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"


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
def get_retriever(db, k=2):
    return db.as_retriever(search_kwargs={"k": k})


# 4. Query + Retrieve
def retrieve(query, retriever):
    print(f"\nUser Query: {query}")
    docs = retriever.invoke(query)

    # print("\n--- Context ---")
    # for i, doc in enumerate(docs, 1):
    #     print(f"\nDocument {i}:")
    #     print(doc.page_content)

    return docs

#5. Generate Answer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])[:1000]

    prompt = f"""
    Answer in one short sentence.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(**inputs, max_new_tokens=50)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Query: ", query)
    print("\nGenerated Answer: ", end="")
    print(answer)
    

# 6. Main
def main():
    db = load_vector_store()
    retriever = get_retriever(db)

    query = "How much did Microsoft pay to acquire GitHub?"

    docs = retrieve(query, retriever)   
    generate_answer(query, docs)        
    


if __name__ == "__main__":
    main()