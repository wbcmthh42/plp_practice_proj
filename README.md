# TechPulse Overview

TechPulse is an innovative tool designed to swiftly identify current technology trends and align them with existing academic research. Its primary goal is to empower educators in developing curricula that are both contemporary and academically robust. By utilizing sentiment analysis and information extraction techniques on social media posts, TechPulse bridges the gap between real-time tech insights and relevant academic studies. This approach fosters a dynamic curriculum development ecosystem, enabling educational institutions to create pertinent course content that keeps pace with the rapidly evolving technological landscape.

Please refer to the documentation for more details. This document provides an overview of the various pipelines used in the application. The application leverages several libraries for data processing, model training, and user interface development.

![Overview of Pipelines](./docs/source/_static/Architecture.png)

## Overview of Pipelines

This diagram illustrates the three main pipelines in the application:

1. **Pipeline 1 (Retrieve arXiv Data)**: This pipeline focuses on retrieving relevant research papers from arXiv using their API. It processes the data to extract useful information and summaries. Please refer to `Pipeline 1 - Retrieve ArXiv Data and Build Vector Store for RAG` under the documentation page.

2. **Pipeline 2 (Finetune Model + Evaluate Model)**: This pipeline involves selecting and finetuning sentiment extraction models. It evaluates the models based on their performance and selects the best one for deployment. Please refer to `Pipeline 2 - Finetune Model for Keyword Extraction and Model Evaluation`under the documentation page.

3. **Pipeline 3 (Retrieve Recent Reddit Posts and Extract Sentiments and Tech Keywords)**: This pipeline retrieves recent posts from Reddit, analyzes sentiments, and extracts relevant tech keywords to provide insights into current trends. Please refer to `Pipeline 3 - Retrieve Recent Reddit Posts and Extract Sentiments and Keywords`under the documentation page.

**Note:** 

- A page detailing how the sentiment model was selected is also available in the documents page: Refer to `Select Sentiment Extraction Model + Evaluate Model` in the documentation html page. 

- A user guide for using the Streamlit UI is included in the documents page: Refer to `TechPulse User Interface` in the documentation html page.  

Detailed information about each pipeline will be provided in separate pages.

## Cloning the Repository

Before setting up the conda environment, you need to clone the repository. Run the following command:

```bash
git clone https://github.com/wbcmthh42/plp_practice_proj
```

## Setting Up the Conda Environment

To set up your conda environment, follow these steps to set up a conda environment named 'techpulse' (this name can be changed according to your preference):

1. **Create a new conda environment**:

   ```bash
   conda create --name techpulse python=3.8
   ```

2. **Activate the environment**:

   ```bash
   conda activate techpulse
   ```

3. **Install the required libraries**:
   You can install the necessary libraries by using the `requirements.txt` file. Run the following command:

   ```bash
   pip install -r requirements.txt
   ```

Make sure to follow the setup instructions carefully to ensure all dependencies are installed correctly.