import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#1. Loading documents from the docs directory
def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory with TextLoader autodetect.
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
    )

    try:
        documents = loader.load()
    except Exception as e:
        print(f"Warning: initial batch load failed ({e}). Falling back to per-file resilient load.")
        documents = []
        for filename in sorted(os.listdir(docs_path)):
            if not filename.lower().endswith('.txt'):
                continue
            filepath = os.path.join(docs_path, filename)
            try:
                loader = TextLoader(filepath, encoding='utf-8', autodetect_encoding=True)
                documents.extend(list(loader.lazy_load()))
            except Exception as e2:
                print(f"Warning: could not load {filepath} with TextLoader ({e2}); using replacement fallback")
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                from langchain_core.documents import Document
                documents.append(Document(page_content=text, metadata={'source': filepath}))

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

# 2. chunking the files
def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

# 3. Create embeddings and store in Chroma vector database
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Main Function")
    persistent_directory = "db/chroma_db"
    # 1. Load documents from the docs directory
    documents = load_documents("docs")

    #2. Chunking the files
    print("\nChunking documents...")
    chunks = split_documents(documents)

    # 3: Create vector store
    vectorstore = create_vector_store(chunks, persistent_directory)

    


if __name__ == "__main__":
    main()