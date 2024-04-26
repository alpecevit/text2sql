# FLAN-T5 Base Text to SQL Model

The training repository for the [text2sql model](https://huggingface.co/alpecevit/flan-t5-base-text2sql) that includes the datasets used for evaluation and training, along with scripts for training, evaluation, and inference. You can find individual datasets listed below.

- [SParC](https://yale-lily.github.io/sparc)
- [Spider](https://yale-lily.github.io/spider)
- [CoSQL](https://yale-lily.github.io/cosql)

While the dataset creation scripts will not be shared, the provided datasets can be used directly for training and evaluating the model using the included scripts. The repository contains everything needed to train and evaluate the model.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Run `inference.py` to evaluate individual test dataset samples.

### OR

Consider downloading the model directly from [HuggingFace](https://huggingface.co/alpecevit/flan-t5-base-text2sql) and run on custom inputs.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("alpecevit/flan-t5-base-text2sql")
model = AutoModelForSeq2SeqLM.from_pretrained("alpecevit/flan-t5-base-text2sql")

input_text = """
transform question and schema to SQL query. question: Who are the top 5 most paid employess by first name, last name, and salary ? schema: employee(salary, bdate, dno, ssn, fname, sex, superssn, address, minit, lname), department(dnumber, mgrstartdate, dname, mgrssn), dept_locations(dnumber, dlocation), project(pnumber, dnum, pname, plocation), works_on(pno, hours, essn), dependent(bdate, essn, dependent_name, sex, relationship).
"""

token_input = tokenizer(input_text, return_tensors="pt").input_ids
output = model.generate(token_input, max_new_tokens=128)
query = tokenizer.decode(output[0], skip_special_tokens=True)

print("Predicted Query:", query)
```

*Output:*

```
SELECT fname, lname, salary FROM employee ORDER BY salary DESC LIMIT 5
```

## Evaluation

```
{'rouge1': 0.8740305983060861, 'rouge2': 0.7763397400315798, 'rougeL': 0.8449832130213266, 'rougeLsum': 0.8447120646910007}
```