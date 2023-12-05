from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

def main():
# Load the ELECTRA-small model and tokenizer
    argp = HfArgumentParser(TrainingArguments)
    
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    # Example context and set of questions
    context = "The Apollo program was a series of space missions conducted by NASA between 1961 and 1972."
    questions = [
        "When did the Apollo program start?",
        "What was the Apollo program?",
        "Who conducted the Apollo program?"
    ]

    # Get answers for each question
    for question in questions:
        answer = answer_question(tokenizer,question, context, model)
        print(f"Question: {question}\nAnswer: {answer}\n")



# Function to answer questions
def answer_question(tokenizer,question, context, model):
    # Encode the question-context pair
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert the tokens to the answer string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer



# Function to answer questions
def answer_question(tokenizer,question, context, model):
    # Encode the question-context pair
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert the tokens to the answer string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer

# Example context and set of questions
context = "The Apollo program was a series of space missions conducted by NASA between 1961 and 1972."
questions = [
    "When did the Apollo program start?",
    "What was the Apollo program?",
    "Who conducted the Apollo program?"
]

# Get answers for each question
for question in questions:
    answer = answer_question(question, context)
    print(f"Question: {question}\nAnswer: {answer}\n")
    
def main():
        argp = HfArgumentParser(TrainingArguments)
    
        
if __name__ == "__main__":
    main()
