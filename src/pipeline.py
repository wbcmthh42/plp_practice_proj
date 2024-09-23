import subprocess
import logging

def run_pipeline():
    # Run model_training.py
    logging.info("Running model_training.py")
    subprocess.run(["python", "-m", "src.model_training"], check=True)
    
    # Run evaluation.py
    logging.info("Running evaluation.py")
    subprocess.run(["python", "-m", "src.evaluation"], check=True)

    # Run reddit_keywords.py
    logging.info("Running reddit_keywords.py")
    subprocess.run(["python", "-m", "src.extract_reddit_keywords"], check=True)

    # Run get_trending_keywords.py
    logging.info("Running get_trending_keywords.py")
    subprocess.run(["python", "-m", "src.get_trending_keywords"], check=True)

if __name__ == "__main__":
    '''python -m src.pipeline'''
    run_pipeline()