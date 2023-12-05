from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import argparse
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

def main():
# Load the ELECTRA-small model and tokenizer
    parser = argparse.ArgumentParser(description='Example Python script with arguments.')
    
    parser.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    

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
       
if __name__ == "__main__":
    main()
