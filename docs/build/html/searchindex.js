Search.setIndex({"alltitles": {"1. Date Range Selection:": [[18, "date-range-selection"]], "2. Top Keywords Display:": [[18, "top-keywords-display"]], "3. Keyword Filtering:": [[18, "keyword-filtering"]], "4. Tech Trends Insight:": [[18, "tech-trends-insight"]], "5. Research Paper Details:": [[18, "research-paper-details"]], "ArXiv Database Construction": [[0, "arxiv-database-construction"]], "BART Configuration": [[6, "bart-configuration"]], "BART and T5 Configurations": [[6, "bart-and-t5-configurations"]], "BERT Configuration": [[6, "bert-configuration"]], "Cloning the Repository": [[9, "cloning-the-repository"]], "Combined Method: VADER + DistilBERT:": [[16, "combined-method-vader-distilbert"]], "Conclusion": [[6, "conclusion"], [11, "conclusion"], [18, "conclusion"]], "Configuration File": [[11, "configuration-file"]], "Configuration Instructions": [[18, "configuration-instructions"]], "Configurations": [[6, "configurations"]], "Contents:": [[4, null]], "Dependencies": [[17, "dependencies"]], "DistilBERT:": [[16, "distilbert"]], "Evaluation Process": [[6, "evaluation-process"]], "Example Final Repo Structure (showing only the key sample files)": [[9, "example-final-repo-structure-showing-only-the-key-sample-files"]], "Expected Outputs": [[11, "expected-outputs"]], "File Structure": [[0, "file-structure"]], "Final Decision": [[16, "final-decision"]], "Flowchart on Sentiment Model Selection": [[16, "flowchart-on-sentiment-model-selection"]], "Implementation": [[16, "implementation"]], "Initial Repo Structure": [[9, "initial-repo-structure"]], "Introduction": [[6, "introduction"], [16, "introduction"], [18, "introduction"]], "Key Features": [[17, "key-features"]], "Key Results": [[16, "key-results"]], "Main UI Display": [[18, "main-ui-display"]], "Model Comparison": [[16, "model-comparison"]], "Model Evaluation Criteria": [[16, "model-evaluation-criteria"]], "Model Selection Methodology": [[16, "model-selection-methodology"]], "Module Description": [[17, "module-description"]], "Modules": [[8, null]], "Note": [[9, "note"]], "Note on Downloading the Gemma-2B Model": [[18, "note-on-downloading-the-gemma-2b-model"]], "Overview of Pipelines": [[9, "overview-of-pipelines"]], "Performance Metrics": [[16, "performance-metrics"]], "Pipeline 1 - Retrieve ArXiv Data and Build Vector Store for RAG": [[0, null]], "Pipeline 2 - Finetune Model for Keyword Extraction and Model Evaluation": [[6, null]], "Pipeline 3 - Retrieve Recent Reddit Posts and Extract Sentiments and Keywords": [[11, null]], "Pipeline Execution": [[6, "pipeline-execution"]], "Pipeline Overview": [[11, "pipeline-overview"]], "Pre-trained Models": [[6, "pre-trained-models"]], "Prerequisites": [[11, "prerequisites"]], "Project Structure relevant to this section": [[6, "project-structure-relevant-to-this-section"]], "RAG Vector Store Construction": [[0, "rag-vector-store-construction"], [0, "id1"]], "Reference Notebooks": [[16, "reference-notebooks"]], "Results for DistilBERT": [[16, "id3"]], "Results for DistilBERT + VADER": [[16, "id5"]], "Results for RoBERTa": [[16, "id4"]], "Results for TextBlob": [[16, "id2"]], "Results for VADER": [[16, "id1"]], "RoBERTa:": [[16, "roberta"]], "Running the Pipeline": [[11, "running-the-pipeline"]], "Search Functionality": [[0, "search-functionality"]], "Select Sentiment Extraction Model + Evaluate Model": [[16, null]], "Setting Up the Conda Environment": [[9, "setting-up-the-conda-environment"]], "Summary / Conclusion": [[16, "summary-conclusion"]], "T5 Configuration": [[6, "t5-configuration"]], "TechPulse Documentation": [[4, null]], "TechPulse Overview": [[9, null]], "TechPulse Streamlit POC with Gemma": [[17, "techpulse-streamlit-poc-with-gemma"]], "TechPulse User Interface": [[18, null]], "TextBlob:": [[16, "textblob"]], "Training Pipeline": [[6, "training-pipeline"]], "Training Process": [[6, "training-process"]], "Usage": [[0, "usage"], [6, "usage"], [17, "usage"], [18, "usage"]], "User Interface Data Flowchart Overview": [[18, "user-interface-data-flowchart-overview"]], "VADER:": [[16, "vader"]], "evaluation module": [[1, null]], "extract_reddit_keywords - Use Finetuned Model to Extract Tech Keywords from Reddit Posts": [[2, null]], "infer_pipeline module - Pipeline to Retrieve Reddit Posts and Extract Sentiments and Keywords": [[5, null]], "model_training module - Finetuning BERT": [[3, null]], "model_training module - Finetuning Bart and T5": [[7, null]], "pipeline_model_finetuning_evaluation module": [[10, null]], "rag module": [[12, null]], "retrieve_papers_with_link module": [[13, null]], "scrape_reddit - Extract from Reddit Posts using PRAW API": [[14, null]], "sentiment_analysis - Extract Sentimentd from Reddit Posts": [[15, null]], "streamlit_poc_with_gemma module": [[17, null]]}, "docnames": ["build_arxiv_vectorstore", "evaluation", "extract_reddit_keywords_with_bart", "finetune_BERT", "index", "infer_pipeline", "model_finetuning_evaluation_pipeline", "model_training", "modules", "overview", "pipeline_model_finetuning_evaluation", "pipeline_retrieve_reddit_post_sentiment_keywords", "rag", "retrieve_papers_with_link", "scrape_reddit", "sentiment_analysis", "sentiment_extraction_model_selection", "streamlit_poc_with_gemma", "techpulse_ui"], "envversion": {"sphinx": 63, "sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1}, "filenames": ["build_arxiv_vectorstore.rst", "evaluation.rst", "extract_reddit_keywords_with_bart.rst", "finetune_BERT.rst", "index.rst", "infer_pipeline.rst", "model_finetuning_evaluation_pipeline.rst", "model_training.rst", "modules.rst", "overview.rst", "pipeline_model_finetuning_evaluation.rst", "pipeline_retrieve_reddit_post_sentiment_keywords.rst", "rag.rst", "retrieve_papers_with_link.rst", "scrape_reddit.rst", "sentiment_analysis.rst", "sentiment_extraction_model_selection.rst", "streamlit_poc_with_gemma.rst", "techpulse_ui.rst"], "indexentries": {"create_keyword_chart() (in module src.streamlit_poc_with_gemma)": [[17, "src.streamlit_poc_with_gemma.create_keyword_chart", false]], "display_keyword_details() (in module src.streamlit_poc_with_gemma)": [[17, "src.streamlit_poc_with_gemma.display_keyword_details", false]], "get_llm_summary() (in module src.streamlit_poc_with_gemma)": [[17, "src.streamlit_poc_with_gemma.get_llm_summary", false]], "get_top_keywords() (in module src.streamlit_poc_with_gemma)": [[17, "src.streamlit_poc_with_gemma.get_top_keywords", false]], "load_and_preprocess_data() (in module src.streamlit_poc_with_gemma)": [[17, "src.streamlit_poc_with_gemma.load_and_preprocess_data", false]], "main() (in module src.streamlit_poc_with_gemma)": [[17, "src.streamlit_poc_with_gemma.main", false]], "module": [[17, "module-src.streamlit_poc_with_gemma", false]], "preprocess_keywords() (in module src.streamlit_poc_with_gemma)": [[17, "src.streamlit_poc_with_gemma.preprocess_keywords", false]], "process_keywords() (in module src.streamlit_poc_with_gemma)": [[17, "src.streamlit_poc_with_gemma.process_keywords", false]], "src.streamlit_poc_with_gemma": [[17, "module-src.streamlit_poc_with_gemma", false]]}, "objects": {"src": [[1, 0, 0, "-", "evaluation"], [2, 0, 0, "-", "extract_reddit_keywords_with_bart"], [3, 0, 0, "-", "finetune_BERT"], [5, 0, 0, "-", "infer_pipeline"], [7, 0, 0, "-", "model_training"], [10, 0, 0, "-", "pipeline_model_finetuning_evaluation"], [12, 0, 0, "-", "rag"], [13, 0, 0, "-", "retrieve_papers_with_link"], [14, 0, 0, "-", "scrape_reddit"], [15, 0, 0, "-", "sentiment_analysis"], [17, 0, 0, "-", "streamlit_poc_with_gemma"]], "src.evaluation": [[1, 1, 1, "", "evaluate"], [1, 1, 1, "", "evaluate_bert"], [1, 1, 1, "", "get_dataset_average_score"], [1, 1, 1, "", "main"]], "src.extract_reddit_keywords_with_bart": [[2, 1, 1, "", "get_keywords"], [2, 1, 1, "", "load_reddit_csv_to_datasets"], [2, 1, 1, "", "main"], [2, 1, 1, "", "save_to_csv"]], "src.finetune_BERT": [[3, 2, 1, "", "BERTFineTuner"], [3, 2, 1, "", "CustomDataset"], [3, 1, 1, "", "main"]], "src.finetune_BERT.BERTFineTuner": [[3, 3, 1, "", "MAX_LEN"], [3, 3, 1, "", "batch_size"], [3, 4, 1, "", "create_dataloaders"], [3, 4, 1, "", "create_datasets"], [3, 3, 1, "", "device"], [3, 4, 1, "", "format_time"], [3, 3, 1, "", "ids_to_labels"], [3, 4, 1, "", "label_keywords"], [3, 3, 1, "", "labels_to_ids"], [3, 3, 1, "", "learning_rate"], [3, 4, 1, "", "load_datasets"], [3, 3, 1, "", "max_len"], [3, 3, 1, "", "model"], [3, 3, 1, "", "num_epochs"], [3, 3, 1, "", "output_dir"], [3, 4, 1, "", "preprocess_text"], [3, 3, 1, "", "pretrained_model"], [3, 4, 1, "", "process_dataset"], [3, 4, 1, "", "process_datasets"], [3, 4, 1, "", "run"], [3, 4, 1, "", "save_model"], [3, 4, 1, "", "setup_device"], [3, 4, 1, "", "setup_tokenizer_and_model"], [3, 4, 1, "", "setup_training"], [3, 3, 1, "", "tokenizer"], [3, 4, 1, "", "train"]], "src.finetune_BERT.CustomDataset": [[3, 3, 1, "", "data"], [3, 3, 1, "", "labels_to_ids"], [3, 3, 1, "", "len"], [3, 3, 1, "", "max_len"], [3, 3, 1, "", "tokenizer"]], "src.infer_pipeline": [[5, 1, 1, "", "run_pipeline"]], "src.model_training": [[7, 1, 1, "", "get_feature"], [7, 1, 1, "", "load_data"], [7, 1, 1, "", "load_model"], [7, 1, 1, "", "main"], [7, 1, 1, "", "train_model"]], "src.pipeline_model_finetuning_evaluation": [[10, 1, 1, "", "run_pipeline"]], "src.rag": [[12, 1, 1, "", "check_vector_store_size"], [12, 1, 1, "", "embed_and_store_documents"], [12, 1, 1, "", "flatten"], [12, 1, 1, "", "get_embeddings"], [12, 1, 1, "", "hybrid_search"], [12, 1, 1, "", "initialize_data_and_model"], [12, 1, 1, "", "keyword_search"], [12, 1, 1, "", "load_vector_store"], [12, 1, 1, "", "main"], [12, 1, 1, "", "setup_rag"], [12, 1, 1, "", "truncate_summary"], [12, 1, 1, "", "vector_search"]], "src.retrieve_papers_with_link": [[13, 1, 1, "", "main"], [13, 1, 1, "", "retrieve_papers_with_link"], [13, 1, 1, "", "save_to_csv"]], "src.scrape_reddit": [[14, 1, 1, "", "fetch_comments_from_submission"], [14, 1, 1, "", "main"], [14, 1, 1, "", "retrieve_list_of_submission_id"], [14, 1, 1, "", "scrape_reddit_comments"]], "src.sentiment_analysis": [[15, 1, 1, "", "generate_wordcloud"], [15, 1, 1, "", "get_sentiment"], [15, 1, 1, "", "get_sentiment_label"], [15, 1, 1, "", "hybrid_sentiment"], [15, 1, 1, "", "main"], [15, 1, 1, "", "preprocess_text"], [15, 1, 1, "", "process_sentiment_analysis"], [15, 1, 1, "", "truncate_to_max_length"]], "src.streamlit_poc_with_gemma": [[17, 1, 1, "", "create_keyword_chart"], [17, 1, 1, "", "display_keyword_details"], [17, 1, 1, "", "get_llm_summary"], [17, 1, 1, "", "get_top_keywords"], [17, 1, 1, "", "load_and_preprocess_data"], [17, 1, 1, "", "main"], [17, 1, 1, "", "preprocess_keywords"], [17, 1, 1, "", "process_keywords"]]}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"], "2": ["py", "class", "Python class"], "3": ["py", "attribute", "Python attribute"], "4": ["py", "method", "Python method"]}, "objtypes": {"0": "py:module", "1": "py:function", "2": "py:class", "3": "py:attribute", "4": "py:method"}, "terms": {"": [1, 3, 6, 16, 18], "0": 6, "000": [0, 16], "00000": 6, "00001": 6, "01": 6, "02": 16, "03": 16, "05": 16, "07": 16, "08": 6, "09": [6, 9, 16], "1": [4, 6, 9, 10, 16, 17], "10": [6, 17, 18], "100": 0, "100k": 9, "12": 9, "13": 16, "14": 16, "15": [0, 12, 16], "16": [6, 9, 16], "16f41d05": 9, "17": 16, "175206f8e788": 9, "1e6": 6, "2": [4, 9, 10, 11, 16], "200": 0, "2024": [6, 9], "22": 16, "25": 16, "256": 6, "28": 16, "29": 16, "3": [4, 6, 9, 16], "30": [0, 12, 16], "32": [6, 16], "33": 16, "336": 9, "35": [9, 16], "36": 16, "3e": 6, "4": 6, "40": 16, "404d": 9, "41": 16, "42": 16, "43": 16, "44": 16, "46": 16, "47": [9, 16], "48": [6, 16], "49": 16, "4o": 16, "5": [6, 12, 17], "50": [6, 16, 18], "500": 6, "51": 16, "52": 16, "53": 16, "55": [6, 16], "56": 16, "57": 16, "58": 16, "59": 16, "60": 16, "600": 16, "61": 16, "63": 16, "64": 16, "65": 16, "67": 16, "68": 16, "69": 16, "7": 16, "70": 16, "71": 16, "72": 16, "73": 16, "74": 16, "75": 16, "76": 16, "78": 16, "79": 16, "8": [12, 16], "80": 16, "81": 16, "82": 16, "84": 16, "85": 16, "86": 16, "87a1": 9, "88": 16, "9": 16, "90": 16, "91": 16, "92": 16, "95": 16, "A": [1, 3, 6, 7, 9, 11, 12, 14, 15, 16, 17], "As": 16, "By": [6, 9, 11, 16, 18], "For": [6, 16, 18], "If": [12, 15, 18], "In": 16, "It": [0, 1, 3, 5, 7, 9, 12, 14, 15, 16], "Its": 9, "The": [0, 1, 2, 3, 5, 6, 9, 11, 12, 14, 15, 16, 17, 18], "These": [6, 16], "To": [0, 3, 6, 9, 11, 17, 18], "_lrschedul": 3, "abil": [6, 16], "abl": [11, 18], "about": [9, 16, 18], "academ": 9, "access": 18, "accord": 9, "account": [16, 18], "accur": 16, "accuraci": [0, 3, 16], "achiev": 16, "across": 16, "activ": 9, "actual": 6, "adjust": [0, 6, 9], "after": [6, 18], "against": 16, "aggreg": 16, "aim": [6, 16, 17], "align": 9, "all": [0, 1, 9, 11, 12, 14, 16], "allow": [0, 3, 6, 9, 12, 17, 18], "along": 0, "alreadi": [6, 12], "also": [0, 1, 7, 9, 16, 18], "altern": [0, 6], "among": 16, "an": [0, 6, 9, 12, 18], "analysi": [5, 9, 11, 15, 16, 17, 18], "analyz": [5, 9, 11, 15, 16, 17, 18], "ani": 18, "annot": 3, "anoth": 15, "anyth": 12, "api": [4, 8, 9, 11, 16, 17], "app": [17, 18], "appli": [3, 16], "applic": [9, 11, 16, 17, 18], "approach": [6, 9, 16], "appropri": 3, "approv": 18, "approxim": [16, 18], "ar": [0, 1, 2, 3, 5, 6, 9, 11, 12, 15, 16, 18], "architectur": 16, "archiv": 0, "argument": [2, 6, 12], "around": 16, "arxiv": [4, 9, 12, 13, 17, 18], "arxiv_papers_2022_2024_with_links_fin": [0, 9], "assess": [6, 16], "assign": 15, "assist": 17, "associ": 12, "assum": 12, "assur": 16, "attention_mask": 7, "augment": [0, 12], "authent": 18, "auto": [2, 6], "automat": 3, "automodelforseq2seqlm": 7, "autotoken": 7, "avail": [3, 9, 16], "averag": 1, "avoid": 16, "awar": 18, "back": 17, "balanc": 16, "bar": 17, "bart": [2, 4, 5, 8, 11], "bart_base_model_result": 6, "bart_tech_keywords_model": [6, 9], "base": [0, 3, 6, 9, 11, 12, 13, 15, 16, 17, 18], "base_model_nam": 6, "batch": [0, 2, 3, 6, 7, 12, 16], "batch_siz": [2, 3, 6], "becaus": [6, 16], "been": [6, 9, 12, 18], "befor": [9, 11, 18], "began": 16, "being": 5, "below": [0, 6, 9, 11, 16], "bert": [1, 4, 8], "bertfinetun": [3, 8], "bertfortokenclassif": 3, "berttokenizerfast": 3, "best": [6, 9, 16], "better": 16, "between": [9, 16, 17], "bidirect": [2, 3, 6], "bin": 9, "both": [0, 6, 9, 12, 15, 16], "bridg": 9, "browser": 18, "build": [4, 9], "built": [0, 17, 18], "c9ac": 9, "calcul": [1, 3, 16], "call": [5, 12, 14, 16], "can": [0, 6, 9, 16, 18], "cannot": 12, "captur": [6, 16], "carefulli": 9, "case": [6, 16, 17], "categori": [0, 12, 13, 15], "certain": 16, "cfg": [1, 2, 3, 6, 7, 12, 13, 14, 15, 17, 18], "chang": 9, "charact": 17, "characterai": 11, "characterist": 16, "chart": 17, "chatgpt": 11, "check": [0, 12, 16, 18], "check_vector_store_s": [8, 12], "checkpoint": [6, 9], "choos": [0, 6, 18], "chosen": 16, "chroma": [0, 9, 12], "class": [3, 16], "classif": [3, 15, 16], "classifi": 16, "clean": 15, "cli": 18, "click": 18, "client": 12, "client_id": 11, "clone": 4, "cloud": 15, "code": [9, 18], "collect": [12, 16], "column": [0, 2, 17], "com": 9, "combin": [0, 6, 12, 15], "command": [2, 6, 9, 11, 18], "comment": [11, 14, 15, 16], "common": [15, 17], "compar": [1, 16], "comparison": 4, "compil": 16, "complet": 6, "complex": [6, 16], "compon": [3, 6], "compound": 15, "compound_scor": 15, "comprehens": [3, 16], "comput": [0, 12, 16], "concaten": 0, "concept": 17, "concis": 17, "conclus": 4, "conda": 4, "conduct": 16, "conf": [0, 6, 9, 11, 18], "confid": 16, "config": [0, 2, 6, 9, 11, 12, 18], "configur": [0, 1, 2, 3, 4, 7, 9, 12, 13, 14, 15, 17], "confus": 16, "conll03": 6, "consid": 16, "consist": [0, 3, 6, 11, 16], "consol": 12, "construct": 4, "contain": [0, 1, 2, 3, 7, 9, 11, 12, 13, 14, 15, 17], "contemporari": 9, "content": [0, 6, 9], "context": 6, "contextu": 6, "contrast": 16, "contribut": 18, "convert": [3, 14, 15, 17], "coordin": 6, "corpora": 6, "correct": [16, 18], "correctli": [6, 9, 11, 16, 18], "cosin": 0, "cost": 16, "count": 17, "cours": [9, 16, 17], "cpu": 3, "creat": [0, 3, 6, 9, 11, 12, 17], "create_dataload": 3, "create_dataset": 3, "create_keyword_chart": [8, 17], "credenti": 11, "criteria": 4, "critic": 16, "csv": [0, 1, 2, 6, 9, 11, 13, 14, 15, 17, 18], "csv_data": 1, "current": 9, "curricula": 9, "curriculum": 9, "custom": [3, 6], "customdataset": [3, 8], "cybersecur": 11, "d_predict": 15, "data": [1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17], "data_level0": 9, "databas": [4, 9, 18], "datafram": [0, 2, 3, 12, 14, 15, 17], "dataload": 3, "datasci": 11, "dataset": [1, 2, 3, 6, 7, 9, 15, 16], "dataset_nam": [1, 6], "datasetdict": 1, "date": [0, 12, 13, 14, 17], "date_rang": 0, "datetim": 14, "dbmdz": 6, "dd": [0, 13, 14], "decai": 6, "decis": 4, "decod": 15, "dedupl": 12, "deep": [6, 16], "default": [0, 12, 16, 18], "defin": [0, 6, 11, 17], "delet": [12, 17], "depend": 9, "deploy": 9, "depth": 16, "deriv": 16, "descript": [2, 3, 5, 6, 8, 14, 15, 16], "design": [2, 3, 9, 14, 16], "desir": 11, "detail": [9, 16], "detect": [0, 3], "determin": 15, "develop": 9, "devic": 3, "df": [12, 15, 17], "diagram": 9, "dict": [3, 7, 15], "dictconfig": [2, 3, 7, 12, 13, 14, 15, 17], "dictionari": [3, 7, 12, 13, 15], "differ": [3, 5, 16], "directli": 6, "directori": [0, 3, 6, 7, 9, 11, 12, 15, 18], "discuss": 17, "disk": [0, 12], "displai": [9, 12, 17], "display_keyword_detail": [8, 17], "distilbert": 11, "distribut": 16, "do": [12, 16, 18], "doc": 9, "document": [0, 6, 9, 11, 12, 16, 18], "doe": 12, "doubl": 18, "due": [6, 16], "duplic": 3, "durat": 3, "dynam": [3, 9], "e": [0, 6, 15, 16], "each": [0, 2, 3, 5, 6, 9, 11, 12, 14, 15, 16, 17], "easi": [0, 3, 6, 9], "ecosystem": 9, "educ": 9, "effect": [6, 16, 18], "effici": [0, 16], "elaps": 3, "element": 12, "emb": 12, "embed": [0, 9, 12], "embed_and_store_docu": [8, 12], "embedding_dim": 12, "embedding_model": 0, "emerg": 17, "empir": 16, "empow": 9, "empti": [12, 17], "enabl": [0, 9], "encod": [3, 6, 7], "encount": 18, "end": [0, 6, 13, 14], "end_dat": 14, "end_date_str": 14, "engag": 17, "english": [6, 11], "enhanc": 16, "ensur": [3, 6, 9, 11, 16, 18], "enter": [12, 18], "entir": [3, 6, 10, 11], "entri": [1, 2, 12, 14, 18], "env": [9, 11], "environ": [4, 18], "epoch": [3, 6], "error": [0, 12, 18], "especi": 16, "essenti": [6, 16], "etc": 9, "eval": [1, 6], "eval_step": 6, "evalu": [3, 4, 8, 9, 10], "evaluate_bert": [1, 8], "evaluation_model_nam": [1, 6], "evaluation_result": 6, "evaluation_strategi": 6, "everi": [0, 6], "evolv": 9, "examin": 16, "exampl": [0, 4, 6, 11, 12, 18], "excel": [6, 16], "execut": [3, 4, 5, 10, 11, 17], "exist": [0, 9, 12], "exit": 12, "expect": [4, 16, 18], "explor": [9, 16, 18], "extend": 3, "extract": [0, 4, 8, 9, 17], "extract_reddit_keyword": [4, 8], "extract_reddit_keywords_with_bart": [2, 5, 9, 11], "extractor_finetuned_bart": [6, 9], "extractor_finetuned_t5": 6, "f": 9, "f1": [1, 6, 16], "face": [2, 6, 7, 12, 15, 18], "facebook": 6, "failur": 0, "fals": 16, "featur": [1, 3, 7, 18], "fetch": [0, 14], "fetch_comments_from_submiss": [8, 14], "field": [0, 1], "file": [1, 2, 3, 4, 6, 13, 14, 15, 17, 18], "file_nam": 14, "file_path": 17, "filenam": 13, "fill": 18, "filter": [3, 14, 15, 17], "final": [4, 15], "find": [0, 18], "fine": [3, 6, 9, 16], "finetun": [4, 8, 9, 11], "finetune_bert": [3, 6, 9], "finetune_llm": 9, "first": 18, "flan": 6, "flatten": [8, 12], "flexibl": 3, "float": [3, 15], "flow": 5, "flowchart": 4, "focus": [9, 16], "folder": 9, "follow": [0, 1, 2, 5, 6, 9, 11, 12, 16, 18], "form": 18, "format": [0, 3, 7, 14], "format_tim": 3, "foster": 9, "found": [0, 12], "foundat": 0, "four": 16, "frame": 6, "framework": 6, "frequenc": 17, "from": [0, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18], "from_dat": [0, 13], "function": [1, 3, 4, 6, 7, 10, 11, 12, 13, 14, 15, 17], "futur": [0, 3], "g": [0, 6, 15, 16], "gap": 9, "gate": 18, "gather": 16, "gener": [0, 1, 2, 6, 11, 12, 15, 16, 17, 18], "generate_wordcloud": [8, 15], "generation_config": 9, "get": [1, 7, 17], "get_dataset_average_scor": [1, 8], "get_embed": [8, 12], "get_featur": [6, 7, 8], "get_keyword": [2, 8], "get_llm_summari": [8, 17], "get_senti": [8, 15], "get_sentiment_label": [8, 15], "get_top_keyword": [8, 17], "git": 9, "github": 9, "give": 16, "given": [2, 12, 13, 14, 15, 16, 17], "global": 12, "go": 18, "goal": 9, "good": 16, "googl": [6, 18], "gpt": 16, "gpu": 3, "gradient_accumulation_step": 6, "grant": 18, "greater": 16, "ground": 1, "guid": [9, 18], "guidelin": 9, "ha": [9, 12, 16, 18], "handl": [0, 1, 3, 6], "hardwar": 3, "harvest": 0, "have": [6, 11, 18], "header": 9, "help": [6, 16], "here": [6, 9, 16], "hf": 6, "hh": 3, "high": 16, "higher": 16, "highest": 16, "highlight": 16, "hold": 9, "horizont": 17, "hour": 18, "how": [6, 9, 11, 16, 18], "html": 9, "http": 9, "hub": [6, 7], "hug": [2, 6, 7, 12, 15, 18], "huggingfac": 18, "human": 16, "hybrid": [0, 12], "hybrid_search": [0, 8, 12], "hybrid_senti": [8, 15], "hydra": [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 18], "i": [0, 1, 2, 3, 5, 6, 9, 11, 12, 14, 15, 16, 18], "id": [3, 14], "ideal": 6, "identif": 16, "identifi": [6, 9, 16], "ids_to_label": 3, "illustr": 9, "ilsilfverskiold": 6, "imag": 15, "imbal": 16, "implement": [0, 3, 4, 6, 17], "import": [6, 16], "improv": [0, 16], "includ": [3, 5, 6, 9, 10, 12, 16, 17, 18], "inconsist": 0, "increas": 16, "increment": 0, "index": 9, "index_metadata": 9, "indic": [12, 16], "individu": [6, 17], "infer": 9, "infer_pipelin": [4, 8, 9, 11], "inference_row_limit": 2, "inform": [0, 6, 9, 16, 18], "initi": [0, 3, 4, 6, 12, 16], "initialize_data_and_model": [8, 12], "innov": 9, "input": [3, 6, 11, 12, 15, 18], "input_fil": 11, "input_id": 7, "insight": [9, 17], "instal": [9, 11], "instanc": 16, "institut": 9, "instruct": [4, 9, 11], "int": [2, 3, 12, 14, 15, 17], "integ": 13, "integr": 3, "intend": 18, "interact": [9, 17], "interest": 16, "interfac": [4, 9], "introduct": [4, 9], "involv": [0, 9, 16, 18], "ipynb": [9, 16], "issu": 18, "item": [0, 12], "its": [6, 12, 16], "join": 17, "json": 9, "jupyt": 9, "keep": 9, "kei": [3, 4, 7, 11], "keyword": [0, 3, 4, 7, 8, 9, 12, 17], "keyword_search": [8, 12], "known": 16, "l6": 0, "label": [3, 7, 11, 15, 16], "label_keyword": 3, "labels_to_id": 3, "landscap": 9, "languag": [0, 6, 9, 17], "larg": [6, 17], "last": 12, "latest": 18, "launch": [0, 17, 18], "learn": [3, 6, 16], "learning_r": [3, 6], "len": [3, 12], "length": [3, 6, 9, 15], "level": 16, "leverag": [9, 16], "librari": [6, 7, 9, 11, 14], "like": [6, 9, 16, 18], "limit": [12, 14], "line": 2, "link": [0, 12, 13], "link_list": 9, "list": [3, 9, 11, 12, 13, 14, 17, 18], "literatur": 9, "llm": [12, 17], "load": [0, 1, 2, 3, 6, 7, 9, 12, 15, 17, 18], "load_and_preprocess_data": [8, 17], "load_data": [6, 7, 8], "load_dataset": 3, "load_model": [6, 7, 8], "load_reddit_csv_to_dataset": [2, 8], "load_vector_stor": [8, 12], "loader": 3, "local": 18, "locat": [2, 18], "log": [0, 1, 5, 9], "logging_step": 6, "logic": 0, "login": 18, "look": [9, 16, 18], "loop": [3, 16], "loss": [0, 3], "lower": [16, 17], "lowercas": 15, "lr_schedul": 3, "m": [0, 1, 6, 7, 11], "machinelearn": 11, "made": 16, "mai": [16, 18], "main": [1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17], "maintain": [16, 18], "make": [9, 16], "makefil": 9, "manag": [0, 3, 5, 6], "mani": 16, "map": 3, "match": [0, 12], "materi": [17, 18], "matric": 16, "matrix": 16, "max_len": [3, 6], "max_length": 15, "max_result": [0, 13], "maximum": [0, 2, 3, 6, 12, 13, 15, 18], "md": 9, "measur": 16, "media": 9, "memori": 12, "mention": 16, "merg": 9, "messag": 12, "metadata": [0, 12], "method": 3, "methodologi": 4, "metric": [3, 4, 6], "mini": 16, "minilm": 0, "misclassifi": 16, "mislead": 16, "mm": [0, 3, 13, 14], "model": [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 15, 17], "model_checkpoint": 6, "model_nam": [1, 2, 6, 7, 11], "model_train": [4, 6, 8, 9], "modifi": 9, "modul": [2, 4, 14, 15], "monitor": [0, 16], "more": [9, 16, 18], "most": [0, 16], "mot": 18, "multipl": 5, "my_mtech": [6, 18], "n": [12, 17], "name": [0, 1, 2, 3, 6, 7, 9, 11, 14, 15, 18], "navig": 18, "necessari": [11, 16], "need": [6, 9, 11, 16, 18], "neg": [15, 16, 17, 18], "network": 0, "neu": 15, "neutral": [15, 16, 18], "new": [2, 9, 12, 16, 17], "nlp": 6, "non": 17, "none": [7, 10, 12, 13, 14, 15, 17], "notebook": 9, "noth": 12, "nuanc": 16, "num_epoch": [3, 6], "num_train_epoch": 6, "number": [0, 2, 3, 6, 12, 13, 14, 17, 18], "oai": 0, "object": [1, 2, 3, 12, 14, 15], "observ": 16, "obtain": 11, "occur": 12, "offer": [6, 16], "omegaconf": [1, 7, 17], "onc": [9, 16, 18], "one": [9, 16], "onli": [4, 17], "open": [0, 18], "openai": 16, "openai_sentiment_label": 16, "optim": [3, 6], "option": [12, 13, 18], "orchestr": [10, 15], "origin": 12, "other": [3, 9, 16], "otherwis": 15, "our": [6, 16], "out": [16, 18], "outlin": [0, 6], "output": [4, 6, 9, 13, 14, 15], "output_dir": [3, 6, 7, 15], "output_fil": [0, 11, 15], "overal": 16, "overview": [4, 6], "p": 1, "pace": 9, "packag": 9, "page": [9, 18], "panda": [0, 3, 12, 15, 17], "paper": [0, 9, 12, 13, 17], "paramet": [0, 1, 2, 3, 6, 7, 9, 12, 13, 14, 15, 17], "paramount": 16, "parquet": 6, "pars": 0, "particularli": 6, "path": [0, 1, 2, 3, 6, 15, 17, 18], "pattern": 16, "pd": [2, 14, 17], "per": 14, "per_device_eval_batch_s": 6, "per_device_train_batch_s": 6, "perform": [0, 1, 4, 5, 6, 9, 11, 12, 15, 18], "persist": [0, 12], "persist_directori": [0, 12], "pertin": 9, "pickl": 9, "pipelin": [2, 4, 8, 10, 12, 15, 16], "pipeline_model_finetuning_evalu": [4, 6, 8, 9], "place": 18, "pleas": [9, 18], "plotli": 17, "plp": [6, 18], "plp_practice_proj": [0, 6, 9, 18], "pmh": 0, "po": 15, "point": [1, 2, 12, 14, 18], "popul": [0, 12], "posit": [15, 16, 18], "post": [4, 6, 8, 9, 16, 17, 18], "potenti": [16, 17], "power": [0, 6], "praw": [4, 5, 8, 11], "praw_output": 11, "pre": [0, 4, 7, 12], "precis": [6, 16], "predict": [15, 16], "prefer": [9, 16, 18], "prepar": [6, 11, 16, 18], "preprocess": [2, 3, 15, 17], "preprocess_keyword": [8, 17], "preprocess_text": [3, 8, 15], "prerequisit": 4, "pretrain": [3, 6], "pretrained_model": [3, 6], "pretrainedtoken": 15, "preval": 16, "prevent": 0, "primari": 9, "print": 12, "problem": 6, "process": [0, 2, 3, 4, 5, 9, 11, 15, 16, 17], "process_dataset": 3, "process_keyword": [8, 17], "process_sentiment_analysi": [8, 15], "program": 3, "progress": 5, "project": [0, 4, 5, 9, 18], "project_root": [0, 6], "promin": 16, "prompt": [12, 18], "proof": 17, "propos": [16, 17], "protocol": 0, "provid": [0, 1, 3, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18], "publicli": 16, "punctuat": [3, 15, 17], "px": 17, "py": [0, 2, 5, 6, 9, 11, 14, 15, 17, 18], "python": [0, 1, 5, 6, 7, 9, 11, 14, 17], "pytorch": 3, "qualiti": 16, "queri": [0, 12], "r": 1, "rag": [4, 8, 9, 17, 18], "rang": [0, 13, 14, 17], "rapidli": 9, "rate": [3, 6, 16], "ratio": 16, "raw": 9, "re": [0, 12], "read": [0, 1, 14], "readabl": 14, "readi": 16, "readm": 9, "real": [9, 16], "recal": [6, 16], "recent": [4, 9, 17], "record": 0, "recreat": 12, "reddit": [4, 6, 8, 9, 16, 17, 18], "reddit_dataset": 11, "reddit_grant_typ": 11, "reddit_infer": 2, "reddit_keyword": 11, "reddit_keywords_for_ui": 9, "reddit_keywords_hybrid": [9, 18], "reddit_keywords_result": [9, 18], "reddit_list": 14, "reddit_password": 11, "reddit_results_fil": 11, "reddit_results_file_for_ui": 18, "reddit_usernam": 11, "reduc": 12, "refer": [9, 18], "reflect": 16, "regress": [2, 6], "relat": [6, 9, 17, 18], "relationship": 6, "relev": [0, 4, 9, 12, 17, 18], "reli": 16, "remain": 17, "remov": [3, 15, 17], "replac": 18, "repo": [4, 18], "repositori": [4, 18], "repres": [13, 17], "represent": [3, 6, 15], "request": [0, 18], "requir": [6, 9, 18], "research": [0, 9, 12, 17], "research_pap": 12, "respons": [0, 3], "result": [0, 1, 2, 4, 6, 9, 11, 12, 13, 14, 15, 17, 18], "results_fil": [1, 6], "retrain": 16, "retri": 0, "retriev": [4, 8, 9, 12, 13, 14, 17], "retrieve_list_of_submission_id": [8, 14], "retrieve_papers_with_link": [0, 4, 8, 9], "return": [0, 2, 3, 7, 12, 13, 14, 15, 17], "review": 16, "robust": 9, "routin": 3, "row": [2, 3, 15, 17], "rule": 16, "run": [0, 1, 3, 4, 5, 6, 9, 10, 17, 18], "run_pipelin": [5, 8, 10], "safetensor": 9, "same": [11, 15, 16], "sampl": [4, 16], "save": [0, 2, 3, 6, 7, 9, 12, 13, 14, 15], "save_model": 3, "save_model_nam": [6, 7], "save_step": 6, "save_to_csv": [2, 8, 13], "scenario": 16, "schedul": 3, "scienc": 0, "score": [1, 6, 11, 15, 16], "scrape": [5, 11, 14], "scrape_reddit": [4, 5, 8, 9, 11], "scrape_reddit_com": [8, 14], "script": [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17], "search": [4, 12, 17], "second": 3, "secret_kei": 11, "section": [0, 4, 11, 16], "see": 18, "select": [4, 6, 9, 17], "semant": 0, "sentenc": [0, 15, 16], "sentiment": [4, 8, 9, 15, 17, 18], "sentiment_analysi": [4, 5, 8, 9, 11], "sentiment_analysis_output": 11, "sentiment_analysis_results_distillbert": 9, "sentiment_analysis_results_roberta": 9, "sentiment_analysis_scoring_distilbert": [9, 16], "sentiment_analysis_scoring_roberta": [9, 16], "sentiment_analysis_textblob_sentlevel": [9, 16], "sentiment_analysis_vader_sentlevel": [9, 16], "sentiment_analyz": 15, "sentiment_by_vader_post_level": 9, "sentiment_by_vader_sentlevel": 9, "sentimentd": [4, 8], "separ": [9, 16], "seq2seq": [1, 7], "sequenc": [3, 6, 7], "sequenti": [5, 6, 11], "seri": [3, 5, 15], "serv": [0, 18], "set": [0, 1, 3, 4, 6, 11, 12, 14, 17, 18], "setup": [3, 9], "setup_devic": 3, "setup_rag": [8, 12], "setup_tokenizer_and_model": 3, "setup_train": 3, "sever": 9, "shape": 12, "shorter": [12, 17], "shot": 15, "should": [1, 3, 6, 11, 18], "show": 4, "showcas": 16, "similar": [0, 6], "simpl": [0, 5], "simpler": 16, "simpli": 17, "simplic": 16, "singl": [3, 6, 10, 12, 17], "size": [2, 3, 6, 12], "smoothli": 6, "social": 9, "sole": 16, "some": 18, "sort": 18, "sourc": [1, 2, 3, 5, 7, 9, 10, 12, 13, 14, 15, 17], "space": 17, "special_tokens_map": 9, "specif": [3, 6, 11, 15, 18], "specifi": [0, 2, 3, 6, 11, 12, 13, 14, 15, 18], "split": [6, 17], "sqlite3": 9, "src": [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18], "ss": 3, "sst": 11, "stack": 17, "stage": 0, "standalon": 3, "start": [0, 13, 14], "start_dat": [0, 14], "start_date_str": 14, "statist": 3, "step": [0, 5, 6, 9, 11, 16, 18], "stop": 17, "stop_word": 17, "stopword": 15, "storag": 0, "store": [4, 9, 11, 12, 18], "str": [1, 2, 3, 7, 12, 14, 15, 17], "straightforward": 16, "streamlin": 6, "streamlit": [0, 9, 18], "streamlit_poc_with_gemma": [0, 4, 8, 18], "strength": 16, "string": [3, 12, 13, 17], "strong": 6, "structur": 4, "student": 17, "studi": 9, "sublist": 12, "submiss": 14, "submission_id": 14, "submit": 18, "subprocess": 5, "subreddit": [11, 14], "subreddit_name_list": 14, "subsequ": 18, "successfulli": 11, "suit": 6, "suitabl": [6, 7, 16], "summar": [2, 6, 16], "summari": [0, 1, 4, 6, 9, 12, 13, 17], "sure": 9, "swiftli": 9, "system": [0, 6, 12, 17], "systemat": 16, "t5": [4, 8], "t5_tech_keywords_model": 6, "take": [2, 12, 18], "taken": 16, "target": [6, 7], "task": [3, 6, 9, 16], "tayjohnni": [6, 18], "team": [16, 18], "tech": [4, 6, 8, 9, 16, 17], "technic": 6, "techniqu": [9, 16], "technolog": 9, "technologi": 9, "temporari": 0, "tensor": 12, "termin": [11, 18], "test": [1, 3, 6], "test_dataset": 3, "test_dataset_process": 3, "text": [0, 1, 3, 6, 7, 12, 15], "than": [12, 16, 17], "thei": [6, 16], "them": [1, 9, 12, 13, 14, 16, 18], "thi": [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "three": [0, 6, 9, 11], "time": [3, 9, 16, 18], "timestamp": 14, "titl": [0, 12, 13], "togeth": [0, 17], "token": [0, 3, 6, 7, 9, 12, 15, 18], "tokenizer_config": 9, "tool": [9, 15, 18], "top": [0, 12, 17], "top_keyword": 17, "top_n": [0, 12], "topic": [6, 17], "torch": [3, 12], "total": 16, "track": 5, "tradit": 0, "train": [0, 1, 3, 4, 7, 9, 10, 12], "train_dataload": 3, "train_dataset": 3, "train_dataset_process": 3, "train_model": [6, 7, 8], "trainer": 6, "training_arg": 9, "transfer": 6, "transform": [0, 2, 3, 6, 7, 15, 16], "treat": 6, "tree": 0, "trend": [5, 9, 17], "true": 16, "truncat": [0, 12, 15], "truncate_summari": [8, 12], "truncate_to_max_length": [8, 15], "truth": 1, "try": 18, "tune": [3, 6, 9, 16], "tuner": 3, "tupl": [3, 7], "two": 15, "txt": 9, "type": [0, 2, 3, 7, 12, 14, 15, 16, 17], "typic": 9, "u": 6, "ui": [0, 9], "uncas": 11, "under": 6, "understand": [6, 16], "unifi": 6, "unselect": 18, "until_d": [0, 13], "unwant": 18, "up": [0, 3, 4, 6, 12, 14, 17, 18], "updat": [0, 2, 12, 13, 18], "url": 15, "us": [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 16, 17, 18], "usag": [1, 3, 4, 7, 9, 12], "user": [0, 4, 6, 9, 12, 17], "util": [3, 6, 9, 16, 17], "v2": 0, "v_predict": 15, "vader": [9, 15], "val_dataload": 3, "val_dataset": 3, "valid": [3, 6], "validation_dataset": 3, "validation_dataset_process": 3, "valu": 2, "vari": 16, "variabl": 12, "variou": [9, 16], "vector": [4, 9, 12, 18], "vector_search": [8, 12], "vector_stor": [0, 9], "verbos": 9, "verifi": [0, 18], "view": [16, 18], "vocab": 9, "wa": [9, 12], "wait": 18, "want": 18, "warmup_step": 6, "wbcmthh42": 9, "we": [6, 16], "weak": 16, "web": [6, 18], "weight": 6, "weight_decai": 6, "well": [6, 16], "were": 16, "when": [1, 6, 16, 18], "where": [0, 1, 6, 12, 15, 16, 18], "which": [0, 6, 9, 10, 16], "while": [12, 16, 18], "wish": 6, "within": [0, 14, 18], "without": [9, 12, 18], "word": [0, 3, 12, 15, 17, 18], "word_limit": [0, 12], "workflow": [6, 11], "world": 16, "would": 17, "wrapper": 14, "written": 1, "xml": 0, "yaml": [0, 6, 9, 11, 18], "yml": 9, "you": [0, 6, 9, 11, 18], "your": [6, 9, 11, 18], "yyyi": [0, 13, 14], "zero": 15}, "titles": ["Pipeline 1 - Retrieve ArXiv Data and Build Vector Store for RAG", "evaluation module", "extract_reddit_keywords - Use Finetuned Model to Extract Tech Keywords from Reddit Posts", "model_training module - Finetuning BERT", "TechPulse Documentation", "infer_pipeline module - Pipeline to Retrieve Reddit Posts and Extract Sentiments and Keywords", "Pipeline 2 - Finetune Model for Keyword Extraction and Model Evaluation", "model_training module - Finetuning Bart and T5", "Modules", "TechPulse Overview", "pipeline_model_finetuning_evaluation module", "Pipeline 3 - Retrieve Recent Reddit Posts and Extract Sentiments and Keywords", "rag module", "retrieve_papers_with_link module", "scrape_reddit - Extract from Reddit Posts using PRAW API", "sentiment_analysis - Extract Sentimentd from Reddit Posts", "Select Sentiment Extraction Model + Evaluate Model", "streamlit_poc_with_gemma module", "TechPulse User Interface"], "titleterms": {"1": [0, 18], "2": [6, 18], "2b": 18, "3": [11, 18], "4": 18, "5": 18, "api": 14, "arxiv": 0, "bart": [6, 7], "bert": [3, 6], "build": 0, "clone": 9, "combin": 16, "comparison": 16, "conclus": [6, 11, 16, 18], "conda": 9, "configur": [6, 11, 18], "construct": 0, "content": 4, "criteria": 16, "data": [0, 18], "databas": 0, "date": 18, "decis": 16, "depend": 17, "descript": 17, "detail": 18, "displai": 18, "distilbert": 16, "document": 4, "download": 18, "environ": 9, "evalu": [1, 6, 16], "exampl": 9, "execut": 6, "expect": 11, "extract": [2, 5, 6, 11, 14, 15, 16], "extract_reddit_keyword": 2, "featur": 17, "file": [0, 9, 11], "filter": 18, "final": [9, 16], "finetun": [2, 3, 6, 7], "flowchart": [16, 18], "from": [2, 14, 15], "function": 0, "gemma": [17, 18], "implement": 16, "infer_pipelin": 5, "initi": 9, "insight": 18, "instruct": 18, "interfac": 18, "introduct": [6, 16, 18], "kei": [9, 16, 17], "keyword": [2, 5, 6, 11, 18], "main": 18, "method": 16, "methodologi": 16, "metric": 16, "model": [2, 6, 16, 18], "model_train": [3, 7], "modul": [1, 3, 5, 7, 8, 10, 12, 13, 17], "note": [9, 18], "notebook": 16, "onli": 9, "output": 11, "overview": [9, 11, 18], "paper": 18, "perform": 16, "pipelin": [0, 5, 6, 9, 11], "pipeline_model_finetuning_evalu": 10, "poc": 17, "post": [2, 5, 11, 14, 15], "praw": 14, "pre": 6, "prerequisit": 11, "process": 6, "project": 6, "rag": [0, 12], "rang": 18, "recent": 11, "reddit": [2, 5, 11, 14, 15], "refer": 16, "relev": 6, "repo": 9, "repositori": 9, "research": 18, "result": 16, "retriev": [0, 5, 11], "retrieve_papers_with_link": 13, "roberta": 16, "run": 11, "sampl": 9, "scrape_reddit": 14, "search": 0, "section": 6, "select": [16, 18], "sentiment": [5, 11, 16], "sentiment_analysi": 15, "sentimentd": 15, "set": 9, "show": 9, "store": 0, "streamlit": 17, "streamlit_poc_with_gemma": 17, "structur": [0, 6, 9], "summari": 16, "t5": [6, 7], "tech": [2, 18], "techpuls": [4, 9, 17, 18], "textblob": 16, "thi": 6, "top": 18, "train": 6, "trend": 18, "ui": 18, "up": 9, "us": [2, 14], "usag": [0, 6, 17, 18], "user": 18, "vader": 16, "vector": 0}})