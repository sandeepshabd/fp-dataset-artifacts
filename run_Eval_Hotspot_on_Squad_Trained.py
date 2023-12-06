import datasets
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helper_hotspot import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

NUM_PREPROCESSING_WORKERS = 1



def main():
    argp = HfArgumentParser(TrainingArguments)
    
    argp.add_argument('--model', type=str,
                      default='./trained_model_SQuAD/',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
 
    training_args, args = argp.parse_args_into_dataclasses()
    dataset = datasets.load_dataset("hotpot_qa", "distractor")

    model_class = AutoModelForQuestionAnswering
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    #prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)

    prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)

    """
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
     """
    #if training_args.do_eval:
    eval_dataset = dataset["validation"]

    eval_dataset_featurized = eval_dataset.map(
        prepare_eval_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=eval_dataset.column_names
    )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None

    trainer_class = QuestionAnsweringTrainer
    eval_kwargs['eval_examples'] = eval_dataset
    eval_kwargs['ignore_keys'] = ['type','level','supporting_facts']
    metric = datasets.load_dataset("hotpot_qa", "distractor")
    compute_metrics = lambda eval_preds: metric.compute(
        predictions=eval_preds.predictions, references=eval_preds.label_ids)

    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    
    training_args, args = argp.parse_args_into_dataclasses()
    args.output_dir ="eval_hotspot_onSquad_model"
    training_args.output_dir ="train_hotspot_onSquad_model"
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )

    results = trainer.evaluate(**eval_kwargs)
    

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

    print('Evaluation results:')
    print(results)

    os.makedirs("eval_hotspot_onSquad_model", exist_ok=True)

    with open(os.path.join("eval_hotspot_onSquad_model", 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
        json.dump(results, f)

    with open(os.path.join("eval_hotspot_onSquad_model", 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
        if args.task == 'qa':
            predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
            for example in eval_dataset:
                example_with_prediction = dict(example)
                example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                f.write(json.dumps(example_with_prediction))
                f.write('\n')



if __name__ == "__main__":
    main()
