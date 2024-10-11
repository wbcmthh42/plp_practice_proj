TechPulse Overview
==================

TechPulse is an innovative tool designed to swiftly identify current technology trends and align them with existing academic research. Its primary goal is to empower educators in developing curricula that are both contemporary and academically robust. By utilizing sentiment analysis and information extraction techniques on social media posts, TechPulse bridges the gap between real-time tech insights and relevant academic studies. This approach fosters a dynamic curriculum development ecosystem, enabling educational institutions to create pertinent course content that keeps pace with the rapidly evolving technological landscape.

Please refer to the documentation for more details. This document provides an overview of the various pipelines used in the application. The application leverages several libraries for data processing, model training, and user interface development.

.. image:: source/_static/Architecture.png
   :alt: Overview of Pipelines
   :align: center

Overview of Pipelines
---------------------

This diagram illustrates the three main pipelines in the application:

1. **Pipeline 1 (Retrieve arXiv Data)**: This pipeline focuses on retrieving relevant research papers from arXiv using their API. It processes the data to extract useful information and summaries. Please refer to :doc:`Pipeline 1 - Retrieve ArXiv Data and Build Vector Store for RAG <build_arxiv_vectorstore>`

2. **Pipeline 2 (Finetune Model + Evaluate Model)**: This pipeline involves selecting and finetuning sentiment extraction models. It evaluates the models based on their performance and selects the best one for deployment. Please refer to :doc:`Pipeline 2 - Finetune Model for Keyword Extraction and Model Evaluation <model_finetuning_evaluation_pipeline>`

3. **Pipeline 3 (Retrieve Recent Reddit Posts and Extract Sentiments and Tech Keywords)**: This pipeline retrieves recent posts from Reddit, analyzes sentiments, and extracts relevant tech keywords to provide insights into current trends. Please refer to :doc:`Pipeline 3 - Retrieve Recent Reddit Posts and Extract Sentiments and Keywords <pipeline_retrieve_reddit_post_sentiment_keywords>`

Note
~~~~~

- A page detailing how the sentiment model was selected is also available in this page: :doc:`Select Sentiment Extraction Model + Evaluate Model <sentiment_extraction_model_selection>`. 

- A user guide for using the Streamlit UI is included here: :doc:`TechPulse User Interface <techpulse_ui>`. 

Detailed information about each pipeline will be provided in separate pages.

Cloning the Repository
----------------------

Before setting up the conda environment, you need to clone the repository. Run the following command:

.. code-block:: bash

    git clone https://github.com/wbcmthh42/plp_practice_proj

Initial Repo Structure
----------------------

.. code-block:: text

    .
    ├── README.md
    ├── conf
    │   └── config.yaml
    ├── docs
    │   ├── Makefile
    │   ├── build
    │   │   └── html
    │   │       ├── index.html
    ├── notebook
    │   ├── Sentiment_Analysis_Scoring_Distilbert+Vader.ipynb
    │   ├── Sentiment_Analysis_Scoring_Distilbert.ipynb
    │   ├── Sentiment_Analysis_Scoring_Roberta.ipynb
    │   ├── sentiment_analysis_textblob_sentlevel.ipynb
    │   └── sentiment_analysis_vader_sentlevel.ipynb
    ├── environment.yml
    └── src
        ├── evaluation.py
        ├── extract_reddit_keywords_with_bart.py
        ├── finetune_BERT.py
        ├── infer_pipeline.py
        ├── model_training.py
        ├── pipeline_model_finetuning_evaluation.py
        ├── rag.py
        ├── retrieve_papers_with_link.py
        ├── scrape_reddit.py
        └── sentiment_analysis.py

Setting Up the Conda Environment
--------------------------------

To set up your conda environment, follow these steps to set up a conda environment named 'techpulse' (this name can be changed according to your preference):

1. **Create a new conda environment and install the dependencies**:

.. code-block:: bash

   conda env create -f environment.yml --verbose 

2. **Activate the environment**:

.. code-block:: bash

   conda activate techpulse

Make sure to follow the setup instructions carefully to ensure all dependencies are installed correctly.

Once the pipeline has been run with the data files, model checkpoints and vector database loaded, the structure will look like the below:

Example Final Repo Structure (showing only the key sample files)
----------------------------------------------------------------

.. code-block:: text

    .
    ├── README.md
    ├── conf
    │   └── config.yaml
    ├── data
    │   ├── 100k.csv
    │   ├── sentiment_analysis_results_distillbert.csv
    │   ├── sentiment_analysis_results_roberta.csv
    │   ├── sentiment_by_vader_post_level.csv
    │   └── sentiment_by_vader_sentlevel.csv
    ├── docs
    │   ├── Makefile
    │   ├── build
    │   │   └── html
    │   │       ├── index.html
    ├── finetune_llm
    ├── notebook
    │   ├── Sentiment_Analysis_Scoring_Distilbert+Vader.ipynb
    │   ├── Sentiment_Analysis_Scoring_Distilbert.ipynb
    │   ├── Sentiment_Analysis_Scoring_Roberta.ipynb
    │   ├── sentiment_analysis_textblob_sentlevel.ipynb
    │   └── sentiment_analysis_vader_sentlevel.ipynb
    ├── outputs
    │   ├── 2024-09-16
    │   │   ├── 12-47-35
    │   │   │   ├── bart_tech_keywords_model
    │   │   │   │   └── checkpoint-336
    │   │   │   ├── model_training.log
    │   │   │   └── tech-keywords-extractor_finetuned_bart
    │   │   │       ├── config.json
    │   │   │       ├── generation_config.json
    │   │   │       ├── merges.txt
    │   │   │       ├── model.safetensors
    │   │   │       ├── special_tokens_map.json
    │   │   │       ├── tokenizer.json
    │   │   │       ├── tokenizer_config.json
    │   │   │       ├── training_args.bin
    │   │   │       └── vocab.json
    ├── reddit_keywords_results
    │   ├── reddit_keywords_for_ui.csv
    │   └── reddit_keywords_hybrid.csv
    ├── environment.yml
    ├── src
    │   ├── evaluation.py
    │   ├── extract_reddit_keywords_with_bart.py
    │   ├── finetune_BERT.py
    │   ├── infer_pipeline.py
    │   ├── model_training.py
    │   ├── pipeline_model_finetuning_evaluation.py
    │   ├── rag.py
    │   ├── retrieve_papers_with_link.py
    │   ├── scrape_reddit.py
    │   └── sentiment_analysis.py
    ├── arxiv/
    │   └── arxiv_papers_2022_2024_with_links_final.csv
    └── vector_store
        ├── 16f41d05-c9ac-404d-87a1-175206f8e788
        │   ├── data_level0.bin
        │   ├── header.bin
        │   ├── index_metadata.pickle
        │   ├── length.bin
        │   └── link_lists.bin
        └── chroma.sqlite3

- **README.md**: This file typically contains an introduction to the project, installation instructions, usage guidelines, and other relevant information for users and developers.

- **conf/config.yaml**: This configuration file is used to store settings and parameters for the application, allowing for easy adjustments without modifying the code.

- **data/**: This directory contains datasets used for analysis and model training, including CSV files for sentiment analysis results and raw data.

- **docs/**: This folder holds documentation files, including a Makefile for building the documentation and HTML output.

- **finetune_llm/**: This directory contains files related to the fine-tuning of language models, including saved model checkpoints etc.

- **notebook/**: This folder contains Jupyter notebooks for various sentiment analysis tasks, allowing for interactive data exploration and model evaluation.

- **outputs/**: This is the hydra outputs directory that stores the results of model training and inference, including logs and model checkpoints.

- **reddit_keywords_results/**: This folder contains CSV files with extracted keywords from Reddit posts, which are used for analysis or UI display.

- **environment.yml**: This file lists the Python packages required for the project conda environment.

- **src/**: This directory contains the source code for the application, including scripts for evaluation, data extraction, model training, and sentiment analysis.

- **arxiv/**: This folder contains CSV files with research papers from arXiv, which are used for retrieving relevant academic literature.

- **vector_store/**: This directory contains files related to a vector database used for storing and retrieving embeddings or other vectorized data.
