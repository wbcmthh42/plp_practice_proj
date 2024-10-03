Model Finetuning and Evaluation Pipeline
=========================================

Introduction
------------

This document outlines the model training process for extracting tech keywords from tech-related Reddit posts. We utilize three pre-trained language models: BART, T5, and BERT, which are fine-tuned on our specific task.

Project Structure (may comprised other files)
---------------------------------------------

.. code-block:: text

    project_root/
    ├── src/
    │   ├── evaluation.py
    │   └── model_training.py


Pre-trained Models
------------------

1. BART (Bidirectional and Auto-Regressive Transformers)
   - A sequence-to-sequence model that combines bidirectional and auto-regressive approaches.
   - Suitable for our task due to its strong performance in text generation and summarization.

2. T5 (Text-to-Text Transfer Transformer)
   - A unified framework that treats every NLP task as a text-to-text problem.
   - Ideal for our keyword extraction task as it can be framed as a text-to-text generation problem.

3. BERT (Bidirectional Encoder Representations from Transformers)
   - A powerful language model that excels in understanding context and relationships in text.
   - Well-suited for our task due to its ability to capture deep contextual information from tech-related posts.

These models are particularly suitable for fine-tuning on tech keyword extraction because:
- They have been pre-trained on large corpora, including web text, which likely includes technical content.
- They can capture complex relationships and context in text, essential for identifying relevant tech keywords.
- They can be fine-tuned for sequence-to-sequence tasks, allowing us to generate keywords from input text directly.

Training Pipeline
-----------------

.. image:: source/_static/model_training_pipeline.png
   :alt: Model Training Pipeline
   :width: 100%

The training pipeline consists of the following steps:

1. Data Preparation
   - Load the Tech Keywords Dataset
   - Split into Train, Validation, and Test sets

2. Model Initialization
   - Load pre-trained models (BART, T5, BERT)
   - Initialize tokenizers for each model

3. Fine-tuning
   - Train each model on the training set
   - Validate performance using the validation set
   - Save model checkpoints

4. Evaluation
   - Evaluate model performance on the test set
   - Select the best-performing model based on evaluation metrics

5. Model Selection
   - Choose the best fine-tuned model for keyword extraction

Training Process
----------------

The training process is implemented in the ``src/model_training.py`` file. Here's an overview of the main functions:

- ``load_data(data)``: Loads the dataset using the Hugging Face datasets library.
- ``load_model(model_name)``: Loads a pre-trained model and tokenizer from the Hugging Face model hub.
- ``get_feature(tokenizer, batch)``: Prepares the input data for training by encoding text and target keywords.
- ``train_model(tokenizer, model, dataset, save_model_name, output_dir, cfg)``: Handles the actual training process, including setting up the trainer, training arguments, and saving the model.

The main function uses Hydra for configuration management, allowing easy customization of training parameters.

Usage
-----

To train the model, run the following command:

.. code-block:: bash

    python -m src.model_training

This will start the training process using the configuration specified in the ``conf/config.yaml`` file.

Evaluation
----------

After training, the models are evaluated using the test set. The evaluation process is implemented in the ``src/evaluation.py`` file.

To run the evaluation script, use the following command:

.. code-block:: bash

    python -m src.evaluation

This will execute the evaluation process on the trained models using the test set. The results will be displayed in the console and may also be saved to a file, depending on the configuration in ``conf/config.yaml``.

The evaluation typically includes metrics such as precision, recall, and F1-score for keyword extraction. These metrics help assess how well the models perform in identifying relevant tech keywords from the input text.

Conclusion
----------

This document provides an overview of the model training process for tech keyword extraction. By fine-tuning powerful pre-trained language models on our specific task, we aim to create an effective system for identifying relevant tech keywords from Reddit posts.