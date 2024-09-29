# %%
# !pip install chromadb
# !pip install sentence-transformers

# %%
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import chromadb
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the CSV file
csv_path = '/Users/tayjohnny/Documents/My_MTECH/PLP/plp_practice_proj/arxiv/arxiv_cs_papers_2022_2024_clean.csv'
df = pd.read_csv(csv_path)

# Step 2: Load a pre-trained LLM model for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # A good model for embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Step 3: Initialize Chroma vector store
client = chromadb.Client()
collection = client.create_collection("research_papers")

# %%
# WORKING

from tqdm import tqdm  # Import tqdm for the progress bar

# Step 4: Function to get embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Step 5: Embed title and summary columns
df['combined'] = df['Title'] + ' ' + df['Summary']
combined_texts = df['combined'].tolist()

# Process in batches to avoid memory issues
batch_size = 100  # Adjust batch size as needed

# Use tqdm for progress tracking
with tqdm(total=len(combined_texts), desc="Processing batches", unit="doc") as pbar:
    for i in range(0, len(combined_texts), batch_size):
        batch_texts = combined_texts[i:i + batch_size]
        embeddings = get_embeddings(batch_texts)
        
        # Step 6: Store embeddings and metadata in Chroma
        for idx, embed in enumerate(embeddings):
            global_idx = i + idx  # Ensure we have a global index to avoid out-of-bound errors
            
            if global_idx < len(df):  # Ensure global_idx is within the bounds of the dataframe
                # Add unique IDs and ensure embeddings are converted properly
                collection.add(
                    ids=[f"doc_{global_idx}"],  # Generate unique IDs for each document
                    documents=[batch_texts[idx]],
                    embeddings=embed.cpu().numpy().tolist(),  # Ensure embeddings are lists of floats
                    metadatas={
                        "Title": df['Title'][global_idx],
                        "Summary": df['Summary'][global_idx],
                        "Updated": str(df['Updated'][global_idx]),
                        "Category": df['Category'][global_idx]
                    }
                )
        # Update the progress bar by the batch size
        pbar.update(len(batch_texts))


# %%
# Step 7: Function to perform vector-based search
def vector_search(query, top_n=5):
    query_embedding = get_embeddings([query])
    results = collection.query(
        query_embeddings=query_embedding.cpu().numpy().tolist(),
        n_results=top_n
    )
    # Flatten the metadata list if needed
    vector_results = flatten(results['metadatas'])
    return vector_results

# Flatten function
def flatten(results):
    flat_results = []
    for sublist in results:
        if isinstance(sublist, list):
            flat_results.extend(sublist)
        else:
            flat_results.append(sublist)
    return flat_results

# Step 8: Function to perform keyword-based search
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

# Step 9: Hybrid search that combines vector and keyword search results
def hybrid_search(query, top_n=5):  # Changed top_n to 5
    # Perform vector and keyword search
    vector_results = vector_search(query, top_n)
    keyword_results = keyword_search(query, top_n)

    # Combine results, ensuring no duplicates
    combined_results = []
    seen_titles = set()

    # Add vector search results
    for result in vector_results:
        if isinstance(result, dict) and 'Title' in result:
            if result['Title'] not in seen_titles:
                combined_results.append(result)
                seen_titles.add(result['Title'])

    # Add keyword search results
    for result in keyword_results:
        if result['Title'] not in seen_titles:
            combined_results.append(result)
            seen_titles.add(result['Title'])

    # Limit to top_n results
    return combined_results[:top_n]

# Step 10: Example usage
query = input("Enter a keyword to search: ")
results = hybrid_search(query, top_n=5)  # Changed top_n to 5

# Function to truncate summary to 30 words
def truncate_summary(summary, word_limit=30):
    words = summary.split()
    if len(words) > word_limit:
        return ' '.join(words[:word_limit]) + '...'  # Append '...' to indicate truncation
    return summary

# Print the results with the truncated summary
for result in results:
    title = result.get('Title', 'No Title')
    category = result.get('Category', 'No Category')
    updated = result.get('Updated', 'No Date')
    summary = truncate_summary(result.get('Summary', 'No Summary'))
    
    # Print the formatted results with the truncated summary
    print(f"\nTitle: {title}")
    print(f"Category: {category}")
    print(f"Updated: {updated}")
    print(f"Summary: {summary}\n")



