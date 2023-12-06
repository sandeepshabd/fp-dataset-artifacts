import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helper_hotspot import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
from datasets import load_dataset

NUM_PREPROCESSING_WORKERS = 1



def main():
# Load the ELECTRA-small model and tokenizer
    parser = argparse.ArgumentParser(description='Example Python script with arguments.')
    
    parser.add_argument('--model', type=str,
                      default='./trained_model_SQuAD/',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    
    parser.add_argument('--questions_path', type=str,
                      default='squad_questions.json',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")

    args = parser.parse_args()
    

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    hotpotqa_dataset = load_dataset("hotpot_qa", "distractor")
    
    tokenized_hotpotqa = hotpotqa_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    accuracy = evaluate(tokenized_hotpotqa["validation"], model, tokenizer)
    print(f"Model Accuracy on HotpotQA: {accuracy * 100:.2f}%")
    

    
def preprocess_function(examples, tokenizer):
     
    questions = [q.strip() for q in examples["question"]]
    concatenated_contexts = []
    for context_set in examples["context"]:
        # Each context_set is a list of (title, context) pairs
        full_context = ""
        sentences = context_set["sentences"]
        
        for sentencelist in sentences:
             full_context = " ".join(sentencelist)
        
        concatenated_contexts.append(full_context)
        
    return tokenizer(questions, concatenated_contexts, truncation=True, padding='max_length', max_length=512)

import torch

def evaluate(dataset, model, tokenizer ):
    model.eval()
    correct_predictions = 0
    total_predictions = len(dataset)

    for example in dataset:
        inputs = {
            "input_ids": torch.tensor([example['input_ids']]),
            "attention_mask": torch.tensor([example['attention_mask']])
        }

        with torch.no_grad():
            outputs = model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        pred_answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
        #print(example)
        actual_answer = example["answer"] if example["answer"] else ""
        if pred_answer.strip().lower() == actual_answer.strip().lower():
            correct_predictions += 1

    return correct_predictions / total_predictions

       
if __name__ == "__main__":
    main()