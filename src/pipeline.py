import subprocess

def run_pipeline():
    # Run model_training.py
    subprocess.run(["python", "model_training.py"], check=True)
    
    # Run evaluation.py
    subprocess.run(["python", "evaluation.py"], check=True)
    
    # Run reddit_keywords.py
    subprocess.run(["python", "reddit_keywords.py"], check=True)

    # Run get_trending_keywords.py
    subprocess.run(["python", "get_trending_keywords.py"], check=True)

if __name__ == "__main__":
    '''python -m src.pipeline'''
    run_pipeline()