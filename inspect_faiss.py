import os
from dotenv import load_dotenv # Import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_vertexai import VertexAIEmbeddings

# --- ADD THESE LINES AT THE TOP ---
# Load environment variables from .env file
load_dotenv()
# ----------------------------------

# Assuming these are the same as in your faiss_store.py
FAISS_DIR = "vector_store"
USER_ID = "maria_test_user" # Replace with the actual user_id your chatbot is using
MEMORY_TYPE = "Note" # This is the specific memory type stored in FAISS

def inspect_faiss_index():
    # It's good practice to verify the loaded project ID for debugging
    print(f"Using GOOGLE_CLOUD_PROJECT from .env: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    print(f"Using GOOGLE_APPLICATION_CREDENTIALS from .env: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")


    embeddings = VertexAIEmbeddings(model_name="text-embedding-004") # Ensure this matches your embedding model
    index_path = os.path.join(FAISS_DIR, f"faiss_index_{USER_ID}_{MEMORY_TYPE}")
    print(f"Constructed FAISS index path: {index_path}")

    if not os.path.exists(index_path):
        print(f"FAISS index not found at: {index_path}")
        print("Run your chatbot first to create and populate the FAISS index.")
        return

    try:
        # Load the FAISS index
        faiss_index = FAISS.load_local(folder_path=index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        print(f"Successfully loaded FAISS index from: {index_path}")

        print("\n--- Inspecting FAISS Index (performing searches) ---")

        # Test queries to see what's retrieved
        test_queries = [
            "What is the capital of France?",
            "Tell me about Maria.",
            "What is my favorite color?",
            "Summarize what I learned today.",
            "Anything about hiking?",
            "What's a cool fact I mentioned?",
        ]

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            # Perform a similarity search
            docs_with_scores = faiss_index.similarity_search_with_score(query, k=3)
            if docs_with_scores:
                for doc, score in docs_with_scores:
                    print(f"  - Content: {doc.page_content[:100]}...")
                    print(f"    Metadata: {doc.metadata}")
                    print(f"    Score: {score:.4f}")
            else:
                print("  No relevant documents found.")

    except Exception as e:
        print(f"Error loading or inspecting FAISS index: {e}")

if __name__ == "__main__":
    inspect_faiss_index()