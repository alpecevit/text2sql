import string
import numpy as np
import nltk


class Processor:
    """
    Class for processing data for training and testing.
    """

    def __init__(self, tokenizer, metric, input_max_length, target_max_length):
        """
        Constructor for Processor class.
        Takes in tokenizer, input_max_length, and target_max_length as input.
        """
        self.tokenizer = tokenizer
        self.metric = metric
        self.input_max_length = input_max_length
        self.target_max_length = target_max_length
    
    def __replace_last_character(self, input_str, chars_to_check, new_char):
        """
        Replace the last character of the input string with new character.
        Returns the modified string.
        """
        if input_str and input_str[-1] in chars_to_check:
            return input_str[:-1] + new_char
        else:
            return input_str
    
    def preprocess_function(self, examples):
        """
        Preprocess function for processing data.
        Return processed data as dictionary.
        """
        prefix = "transform question and schema to SQL query."
        inputs = []
        for question, struc in zip(examples['question'], examples['structure']):
            schema = struc + '.'
            question = self.__replace_last_character(
                                                    question, 
                                                    string.punctuation, 
                                                    '?'
                                                    )
            inp = f"{prefix} question: {question} schema: {schema}"
            inputs.append(inp)
        targets = [self.__replace_last_character(query, [";"], "") for query in
                examples['query']]
        model_inputs = self.tokenizer(
                                        inputs,
                                        max_length=self.input_max_length,
                                        truncation=True
                                    )
        labels = self.tokenizer(
                                text_target=targets,
                                max_length=self.target_max_length,
                                truncation=True
                                )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def postrocess_function(self,examples):
        """
        Postprocess function for evaluating data.
        Return processed data as dictionary.
        """
        prefix = "transform question and schema to SQL query."
        inputs = []
        for question, schem in zip(examples['question'], examples['structure']):
            schema = schem + '.'
            question = self.__replace_last_character(
                                                        question,
                                                        string.punctuation,
                                                        '?'
                                                    )
            inp = f"{prefix} question: {question} schema: {schema}"
            inputs.append(inp)
        formatted_inputs = {"text": inputs}
        return formatted_inputs
    
    def compute_metrics(self, eval_pred):
        """
        Method to compute rouge score for evaluation.
        """
        predictions, labels = eval_pred
        predictions = np.where(
                                predictions != -100,
                                predictions,
                                self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(
                                                    predictions,
                                                    skip_special_tokens=True
                                                    )
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
                                            labels,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True
                                            )

        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) 
                         for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                          for label in decoded_labels]

        result = self.metric.compute(
                                    predictions=decoded_preds,
                                    references=decoded_labels,
                                    use_stemmer=True,
                                    use_aggregator=True
                                    )

        result = {key: value * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) 
                           for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}