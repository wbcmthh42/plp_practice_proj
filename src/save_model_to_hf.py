from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Load the model and tokenizer
model_path = "outputs/2024-09-16/08-48-55/tech-keywords-extractor_finetuned_t5"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a dummy trainer (we don't need actual training arguments)
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_tech_keywords_model",
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
)

# Push the model to the Hub
trainer.push_to_hub("wbcmthh42/t5_tech_keywords_model")