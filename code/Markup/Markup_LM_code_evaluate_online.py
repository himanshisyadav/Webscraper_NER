from transformers import AutoProcessor, AutoModelForTokenClassification, get_scheduler, MarkupLMProcessor
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import evaluate
import pdb
import glob
from optimum.bettertransformer import BetterTransformer
import csv
import sklearn


def compute_metrics(metric, return_entity_level_metrics=True):
    results = metric.compute()
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

def main():
    metric = evaluate.load("seqeval")
    with open('preds.csv', newline='') as csvfile:
        preds = list(csv.reader(csvfile))
        # print(preds)
    with open('refs.csv', newline='') as csvfile:
        refs = list(csv.reader(csvfile))
        # print(refs)

    metric.add_batch(predictions=preds, references=refs,)
    eval_metric = compute_metrics(metric)
    print(eval_metric)



if __name__ == "__main__":
    main()
