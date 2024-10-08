"""
Module Description:
This script, `pipeline.py`, is a simple pipeline manager that sequentially runs multiple Python scripts in the project.
It uses subprocess calls to execute different modules, including Reddit scraping, sentiment analysis, keyword extraction, and trending keyword analysis.

Each step in the pipeline is logged to track execution progress.

The scripts being executed are:
1. `scrape_reddit.py` - Scrapes Reddit data.
2. `sentiment_analysis.py` - Analyzes sentiment in the scraped Reddit data.
3. `extract_reddit_keywords_with_bart.py` - Extracts keywords from the Reddit data using the BART model.
"""
import subprocess
import logging

def run_pipeline():
    """
    Runs the data processing pipeline by sequentially executing a series of Python scripts.

    The pipeline executes the following scripts:
    1. `scrape_reddit.py`: Scrapes Reddit data using PRAW.
    2. `sentiment_analysis.py`: Performs sentiment analysis on the scraped Reddit data.
    3. `extract_reddit_keywords_with_bart.py`: Extracts keywords from Reddit posts using the BART model.

    Each script is run using a subprocess, and the process flow is logged for tracking.
    """
    import subprocess
import logging

def run_pipeline():
    
    # Run PRAW retrieval
    logging.info("Running scrape_reddit.py")
    subprocess.run(["python", "-m", "src.scrape_reddit"], check=True)
    
    # # Run PRAW retrieval
    logging.info("Running sentiment_analysis.py")
    subprocess.run(["python", "-m", "src.sentiment_analysis"], check=True)

    # Run reddit_keywords.py
    logging.info("Running reddit_keywords.py")
    subprocess.run(["python", "-m", "src.extract_reddit_keywords_with_bart"], check=True)

if __name__ == "__main__":
    '''python -m src.pipeline'''
    run_pipeline()