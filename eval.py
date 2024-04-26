from transformers import (AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          AutoModelForSeq2SeqLM)

import evaluate
from torch.utils.data import DataLoader
from datasets import Dataset
from processor.Processor import Processor
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

INPUT_MAX_LENGTH = 1024
TARGET_MAX_LENGTH = 128
MODEL_ID = "flan-t5-base-text2sql"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to("mps")
metric = evaluate.load("rouge")
processor = Processor(tokenizer, metric, INPUT_MAX_LENGTH, TARGET_MAX_LENGTH)

# load dataset
data = Dataset.load_from_disk('data/test_data')

# process data
eval_data = data.map(processor.preprocess_function,
                batched=True,
                remove_columns=[
                    'db_id', 'query', 'question', 'structure'
                ])

# create data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                       model=MODEL_ID,
                                       label_pad_token_id=-100
                                       )

# load data into DataLoader for iteration
eval_data = DataLoader(eval_data,
                       batch_size=1,
                       collate_fn=data_collator
                       )

# generate predictions and actual values
preds = []
labels = []
for batch in tqdm(eval_data):
    inputs = batch['input_ids'].to("mps")
    label = batch['labels'].to("mps")

    output = model.generate(inputs, max_new_tokens=processor.target_max_length)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    label = tokenizer.decode(label[0], skip_special_tokens=True)
    preds.append(output)
    labels.append(label)

# rouge score
print(metric.compute(predictions=preds, references=labels))
