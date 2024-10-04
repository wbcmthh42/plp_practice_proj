"""
This module provides a ArXiv Research Paper Retrieval System using a Retrieval-Augmented Generator (RAG) pipeline.
It allows users to search for research papers based on keywords and returns relevant results,
including paper titles, categories, updated dates, summaries, and links.
"""
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from chromadb.errors import InvalidCollectionException
import hydra
from omegaconf import DictConfig
import os

# Global variables
df = None
tokenizer = None
model = None
client = None
collection = None
persist_directory = None

def initialize_data_and_model(cfg):
    """
    Initialize the data and model for the RAG pipeline.

    This function initializes the following objects:

    - df: a Pandas DataFrame containing the input data
    - tokenizer: a Hugging Face tokenizer for the pre-trained LLM
    - model: a Hugging Face model for the pre-trained LLM
    - client: a Chroma vector store client with persistence
    - persist_directory: the directory where the vector store is persisted

    The function takes a Hydra config object as input and sets up the data and model based on the config.

    Returns:
        df, tokenizer, model, client: the initialized objects
    """
    global df, tokenizer, model, client, persist_directory

    # Step 1: Load the CSV file
    df = pd.read_csv(cfg.output_file)

    # Create the 'combined' column
    df['combined'] = df['Title'] + ' ' + df['Summary']

    # Step 2: Load a pre-trained LLM model for embeddings
    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model.name)
    model = AutoModel.from_pretrained(cfg.embedding_model.name)

    # Step 3: Initialize Chroma vector store with persistence
    persist_directory = cfg.vector_store.persist_directory
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)

    print(f"Initialized client with persist_directory: {persist_directory}")
    return df, tokenizer, model, client

# Function to get embeddings
def get_embeddings(texts):
    """
    Compute the embeddings for the given texts.

    Args:
        texts (List[str]): the input texts to embed

    Returns:
        torch.Tensor: the embeddings for the input texts, with shape (len(texts), embedding_dim)
    """
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Function to embed and store documents
def embed_and_store_documents():
    """
    Embed the combined texts (title + summary) and store them in a Chroma vector store.

    This function takes no arguments and returns nothing. It uses the following global variables:

    - df: a Pandas DataFrame containing the research papers
    - client: a Chroma vector store client with persistence
    - persist_directory: the directory where the vector store is persisted

    The function does the following:

    1. Creates a new collection in the Chroma vector store called "research_papers".
    2. Embeds the combined texts (title + summary) in batches using the get_embeddings function.
    3. Stores the embeddings and metadata in the Chroma vector store.
    4. Prints a message to the console indicating that the vector store has been created and saved to disk.

    The embeddings are stored in the vector store with the following metadata:

    - Title: the title of the research paper
    - Summary: the summary of the research paper
    - Updated: the date the research paper was last updated
    - Category: the category of the research paper
    - Link: the link to the research paper

    The function uses a batch size of 8 documents to reduce memory usage.
    """
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
                            "Category": df['Category'][global_idx],
                            "Link": df['Link'][global_idx]
                        }
                    )
            pbar.update(len(batch_texts))

    print("Vector store created and saved to disk")

# Function to load vector store
def load_vector_store():
    """
    Loads the vector store from disk. If the vector store is empty, it will be recreated and populated with the documents in the DataFrame.
    
    This function assumes that the Chroma client is already initialized. If it is not, it will print an error message and return without doing anything.
    
    If an existing collection is found but is empty, it will be deleted and recreated with the documents in the DataFrame. If an error occurs while loading the vector store, it will print an error message and return without doing anything.
    
    Returns:
        None
    """
    global client, collection
    if client is None:
        print("Error: Client is not initialized. Please run initialize_data_and_model first.")
        return

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
    except Exception as e:
        print(f"An error occurred while loading the vector store: {str(e)}")

# Vector search function
def vector_search(query, top_n=5):
    """
    Perform a vector search for a given query in the vector store.

    Args:
        query: The query string to search for.
        top_n: The number of top results to return. Defaults to 5.

    Returns:
        A list of dictionaries containing the metadata associated with the top results.
    """
    query_embedding = get_embeddings([query])
    results = collection.query(
        query_embeddings=query_embedding.cpu().numpy().tolist(),
        n_results=top_n
    )
    return flatten(results['metadatas'])

# Flatten function
def flatten(results):
    """
    Flatten a list of lists into a single list.

    Args:
        results (List[List]): A list of lists to be flattened.

    Returns:
        List: A single list containing all elements from the sublists.
    """
    flat_results = []
    for sublist in results:
        if isinstance(sublist, list):
            flat_results.extend(sublist)
        else:
            flat_results.append(sublist)
    return flat_results

# Keyword search function
def keyword_search(query, top_n=15):
    """
    Perform a keyword search for a given query in the dataframe.

    Args:
        query: The query string to search for.
        top_n: The number of top results to return. Defaults to 15.

    Returns:
        A list of dictionaries containing the metadata associated with the top results.
    """
    keyword_results = []
    for idx, row in df.iterrows():
        if query.lower() in row['combined'].lower():
            keyword_results.append({
                "Title": row['Title'],
                "Updated": str(row['Updated']),
                "Category": row['Category'],
                "Summary": row['Summary'],
                "Link": row['Link']
            })
        if len(keyword_results) >= top_n:
            break
    return keyword_results

# Hybrid search function
def hybrid_search(query, top_n=5):
    """
    Perform a hybrid search for a given query in both the vector store and the dataframe.

    This function combines the results of a vector search and a keyword search, and returns the top
    results from both. The results are deduplicated by title, and the top results are returned.

    Args:
        query (str): The query string to search for.
        top_n (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        list: A list of dictionaries containing the metadata associated with the top results.
    """
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
    """
    Truncate a given summary to a specified word limit.

    Args:
        summary (str): The summary string to truncate.
        word_limit (int, optional): The maximum number of words to allow in the summary. Defaults to 30.

    Returns:
        str: The truncated summary, or the original summary if it was already shorter than the word limit.
    """
    words = summary.split()
    if len(words) > word_limit:
        return ' '.join(words[:word_limit]) + '...'
    return summary

# Add this new function
def check_vector_store_size():
    """
    Check if the number of items in the vector store matches the number of items in the DataFrame.
    If not, re-create the vector store and populate it with the items from the DataFrame.
    """
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

def setup_rag(cfg):
    """
    Set up the RAG pipeline by loading the DataFrame, initializing the model and tokenizer, and loading the vector store.

    Args:
        cfg (DictConfig): The Hydra configuration object.

    Returns:
        None
    """
    global df, tokenizer, model, client, collection, persist_directory
    df, tokenizer, model, client = initialize_data_and_model(cfg)
    if client is not None:
        load_vector_store()
    else:
        print("Failed to initialize client. Vector store not loaded.")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    The main entry point of the script. This function sets up the RAG pipeline, loads the vector store, checks its size, and provides an example usage of the hybrid search function.

    The example usage prompts the user to enter a keyword to search, and then displays the top N results. The results include the title, category, updated date, summary, and link of each paper.

    If the vector store cannot be created or loaded, the function prints an error message and exits.
    """
    setup_rag(cfg)
    global df, tokenizer, model, client, collection, persist_directory

    # Initialize data and model
    df, tokenizer, model, client = initialize_data_and_model(cfg)
    
    if client is None:
        print("Error: Failed to initialize client. Exiting.")
        return

    # Load the vector store (this will create it if it doesn't exist)
    load_vector_store()

    # Check the size of the vector store
    check_vector_store_size()

    # Example usage
    if collection is not None and collection.count() > 0:
        query = input("Enter a keyword to search: ")
        results = hybrid_search(query, top_n=cfg.search.top_n)

        for result in results:
            title = result.get('Title', 'No Title')
            category = result.get('Category', 'No Category')
            updated = result.get('Updated', 'No Date')
            summary = truncate_summary(result.get('Summary', 'No Summary'), word_limit=cfg.summary.word_limit)
            link = result.get('Link', 'No Link')

            print(f"\nTitle: {title}")
            print(f"Category: {category}")
            print(f"Updated: {updated}")
            print(f"Summary: {summary}\n")
            print(f"Link: {link}\n")
    else:
        print("Failed to create or load the vector store. Please check the error messages above.")


if __name__ == "__main__":
    main()