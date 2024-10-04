"""
Module for running the entire pipeline, including model training and evaluation.

This module provides a single function, `run_pipeline`, which orchestrates the execution of the model training and evaluation scripts.

Functions:
    run_pipeline: Runs the entire pipeline, including model training and evaluation.
"""
import subprocess
import logging

def run_pipeline() -> None:
    # Run model_training.py
    """
    Runs the entire pipeline:
    1. Model training
    2. Evaluation
    """
    logging.info("Running model_training.py")
    subprocess.run(["python", "-m", "src.model_training"], check=True)
    
    # Run evaluation.py
    logging.info("Running evaluation.py")
    subprocess.run(["python", "-m", "src.evaluation"], check=True)

if __name__ == "__main__":
    '''python -m src.pipeline_model_finetuning_evaluation'''
    run_pipeline()