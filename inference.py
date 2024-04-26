from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from torch.utils.data import DataLoader
from datasets import Dataset
from processor.Processor import Processor
import random
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
eval_data = data.map(processor.postrocess_function, batched=True)

# extract examples
random_integer = random.randint(0, len(eval_data['text']))
ex_db = eval_data['db_id'][random_integer] 
ex_input = eval_data['text'][random_integer]
ex_query = eval_data['query'][random_integer]
ex_question = eval_data['question'][random_integer]

# tokenize the ex_input
token_input = tokenizer(ex_input, return_tensors="pt").input_ids.to("mps")

# generate SQL query
output = model.generate(token_input, max_new_tokens=processor.target_max_length)

# decode the query
query = tokenizer.decode(output[0], skip_special_tokens=True)

# print the examples and the generated query
print("Database ID:", ex_db)
print("Question:", ex_question)
print("Original Query:", ex_query)
print("Predicted Query:", query)
