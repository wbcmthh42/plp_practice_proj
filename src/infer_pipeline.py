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
    subprocess.run(["python", "-m", "src.extract_reddit_keywords"], check=True)

    # # Run get_trending_keywords.py
    logging.info("Running get_trending_keywords.py")
    subprocess.run(["python", "-m", "src.get_trending_keywords"], check=True)

if __name__ == "__main__":
    '''python -m src.pipeline'''
    run_pipeline()