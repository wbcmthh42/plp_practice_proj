import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from chromadb.errors import InvalidCollectionException

# Step 1: Load the CSV file
csv_path = '/Users/tayjohnny/Documents/My_MTECH/PLP/plp_practice_proj/arxiv/arxiv_cs_papers_2022_2024_clean.csv'
df = pd.read_csv(csv_path)
# df = df[:10000]

# Create the 'combined' column
df['combined'] = df['Title'] + ' ' + df['Summary']

# Step 2: Load a pre-trained LLM model for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Step 3: Initialize Chroma vector store with persistence
persist_directory = "./vector_store"
client = chromadb.PersistentClient(path=persist_directory)

# Function to get embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Function to embed and store documents
def embed_and_store_documents():
    global collection
    collection = client.create_collection("research_papers")
    print("Created new collection")

    combined_texts = df['combined'].tolist()

    batch_size = 8
    with tqdm(total=len(combined_texts), desc="Processing batches", unit="doc") as pbar:
        for i in range(0, len(combined_texts), batch_size):
            batch_texts = combined_texts[i:i + batch_size]
            embeddings = get_embeddings(batch_texts)
            
            for idx, embed in enumerate(embeddings):
                global_idx = i + idx
                if global_idx < len(df):
                    collection.add(
                        ids=[f"doc_{global_idx}"],
                        documents=[batch_texts[idx]],
                        embeddings=embed.cpu().numpy().tolist(),
                        metadatas={
                            "Title": df['Title'][global_idx],
                            "Summary": df['Summary'][global_idx],
                            "Updated": str(df['Updated'][global_idx]),
                            "Category": df['Category'][global_idx]
                        }
                    )
            pbar.update(len(batch_texts))

    print("Vector store created and saved to disk")

# Function to load vector store
def load_vector_store():
    global client, collection
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection("research_papers")
        if collection.count() == 0:
            print("Vector store is empty. Re-creating and populating...")
            client.delete_collection("research_papers")
            embed_and_store_documents()
        else:
            print(f"Loaded vector store from disk with {collection.count()} items")
    except InvalidCollectionException:
        print("No existing collection found. Creating a new one and embedding documents...")
        embed_and_store_documents()

# Vector search function
def vector_search(query, top_n=5):
    query_embedding = get_embeddings([query])
    results = collection.query(
        query_embeddings=query_embedding.cpu().numpy().tolist(),
        n_results=top_n
    )
    return flatten(results['metadatas'])

# Flatten function
def flatten(results):
    flat_results = []
    for sublist in results:
        if isinstance(sublist, list):
            flat_results.extend(sublist)
        else:
            flat_results.append(sublist)
    return flat_results

# Keyword search function
def keyword_search(query, top_n=5):
    keyword_results = []
    for idx, row in df.iterrows():
        if query.lower() in row['combined'].lower():
            keyword_results.append({
                "Title": row['Title'],
                "Updated": str(row['Updated']),
                "Category": row['Category'],
                "Summary": row['Summary']
            })
        if len(keyword_results) >= top_n:
            break
    return keyword_results

# Hybrid search function
def hybrid_search(query, top_n=5):
    vector_results = vector_search(query, top_n)
    keyword_results = keyword_search(query, top_n)

    combined_results = []
    seen_titles = set()

    for result in vector_results:
        if isinstance(result, dict) and 'Title' in result:
            if result['Title'] not in seen_titles:
                combined_results.append(result)
                seen_titles.add(result['Title'])

    for result in keyword_results:
        if result['Title'] not in seen_titles:
            combined_results.append(result)
            seen_titles.add(result['Title'])

    return combined_results[:top_n]

# Function to truncate summary
def truncate_summary(summary, word_limit=30):
    words = summary.split()
    if len(words) > word_limit:
        return ' '.join(words[:word_limit]) + '...'
    return summary

# Add this new function
def check_vector_store_size():
    if collection is not None:
        collection_size = collection.count()
        df_size = len(df)
        print(f"Number of items in vector store: {collection_size}")
        print(f"Number of items in DataFrame: {df_size}")
        if collection_size == df_size:
            print("All items from the DataFrame are stored in the vector store.")
        elif collection_size < df_size:
            print(f"Warning: Only {collection_size} out of {df_size} items are stored in the vector store.")
            print("Re-creating and populating the vector store...")
            client.delete_collection("research_papers")
            embed_and_store_documents()
        else:
            print("Warning: The vector store contains more items than the DataFrame.")
    else:
        print("Vector store is not loaded or created.")

# Main execution
if __name__ == "__main__":
    # Load the vector store (this will create it if it doesn't exist)
    load_vector_store()

    # Check the size of the vector store
    check_vector_store_size()

    # Example usage
    if collection is not None and collection.count() > 0:
        query = input("Enter a keyword to search: ")
        results = hybrid_search(query, top_n=5)

        for result in results:
            title = result.get('Title', 'No Title')
            category = result.get('Category', 'No Category')
            updated = result.get('Updated', 'No Date')
            summary = truncate_summary(result.get('Summary', 'No Summary'))
            
            print(f"\nTitle: {title}")
            print(f"Category: {category}")
            print(f"Updated: {updated}")
            print(f"Summary: {summary}\n")
    else:
        print("Failed to create or load the vector store. Please check the error messages above.")