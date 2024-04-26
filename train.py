from transformers import (AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)
import evaluate
from datasets import Dataset
from processor.Processor import Processor
import warnings

warnings.filterwarnings('ignore')

INPUT_MAX_LENGTH = 1024
TARGET_MAX_LENGTH = 128
MODEL_ID = "google/flan-t5-base"
OUTPUT_MODEL_ID = "flan-t5-base-text2sql"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
rouge = evaluate.load("rouge")
processor = Processor(tokenizer, rouge, INPUT_MAX_LENGTH, TARGET_MAX_LENGTH)


# adding "<" to the tokenizer
new_tokens = ["<"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# load dataset
data = Dataset.load_from_disk('data/train_data')

# process data
data = data.map(processor.preprocess_function,
                batched=True,
                remove_columns=[
                    'db_id', 'query', 'question', 'structure'
                ])

# train and test split
data = data.train_test_split(test_size=0.2, seed=42)

# data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                       model=MODEL_ID,
                                       label_pad_token_id=-100
                                       )

# training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_MODEL_ID + "-checkpoints",
    evaluation_strategy="epoch",
    save_safetensors=False,
    save_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    generation_max_length=processor.target_max_length,
)

# trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics
)

# training
print("Started training:")
trainer.train() # resume_from_checkpoint=True

# saving the model
save_model_name = OUTPUT_MODEL_ID
trainer.model.save_pretrained(save_model_name)
tokenizer.save_pretrained(save_model_name)
