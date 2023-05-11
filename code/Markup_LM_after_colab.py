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
from sadice import SelfAdjDiceLoss

class MarkupLMDataset(Dataset):
    """Dataset for token classification with MarkupLM."""

    def __init__(self, data, processor=None, max_length=512, tokenizer=None):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        print("Length of Data:",len(self.data))
        return len(self.data)

    def __getitem__(self, idx):
        # first, get nodes, xpaths and node labels
        item = self.data[idx]
        # pdb.set_trace()
        nodes, xpaths, node_labels = item['nodes'], item['xpaths'], item['node_labels']
        l = []
        l.append(node_labels)
        
        # # provide to processor
        encoding = self.processor(nodes=nodes.split("!@#$%^&*"), xpaths=xpaths.split("!@#$%^&*"), node_labels=l, padding="max_length", max_length=self.max_length, return_tensors="pt", truncation=True)

        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        return encoding

def get_labels(label_list, predictions, references, device):
    # Transform predictions and references tensos to numpy arrays
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return  true_predictions, true_labels

def compute_metrics(return_entity_level_metrics=True):
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
    import glob
    #Get data from directory ../training_data
    files = glob.glob("../training_data/nodes_xpaths/Labeled_*.csv")

    id2label = {0:'Address', 1:'Beds', 2:'Contact', 3:'NE', 4: 'Name', 5:'Price', 6:'Amenities'}
    label2id = {label:id for id, label in id2label.items()}

    data = []
    nodes = []
    xpaths = []
    node_labels = []
    for file in files:
        nodes_xpaths = pd.read_csv(file,delimiter=",")
        nodes_xpaths = nodes_xpaths.dropna()
        nodes_xpaths['Labels'].replace(['Address', 'Beds', 'Contact', 'NE', 'Name', 'Price', 'Amenities'], [0, 1, 2, 3, 4, 5, 6], inplace=True)
        nodes_xpaths = nodes_xpaths.rename({"nodes": "nodes", "Labels": "node_labels", "xpaths": "xpaths"}, axis = 'columns')
        cols = nodes_xpaths.columns.tolist()
        cols = [cols[0], cols[2], cols[1]]
        nodes_xpaths = nodes_xpaths[cols]
        nodes.append(nodes_xpaths['nodes'].values.tolist())
        xpaths.append(nodes_xpaths['xpaths'].values.tolist())
        node_labels.append(nodes_xpaths['node_labels'].values.tolist())
        # data.append({'nodes': nodes_xpaths['nodes'].values.tolist(), 'xpaths': nodes_xpaths['xpaths'].values.tolist(), 'node_labels': nodes_xpaths['node_labels'].values.tolist()})

    nodes = [item for sublist in nodes for item in sublist]
    xpaths = [item for sublist in xpaths for item in sublist]
    node_labels = [item for sublist in node_labels for item in sublist]

    for i in range(len(nodes)):
    data.append({'nodes': nodes[i], 'xpaths': xpaths[i], 'node_labels': node_labels[i]})

    processor = MarkupLMProcessor.from_pretrained("/project/rcc/hyadav/markuplm-base")
    processor.parse_html = False

    tokenizer = MarkupLMTokenizerFast.from_pretrained("/project/rcc/hyadav/markuplm-base")
    dataset = MarkupLMDataset(data=data, processor=processor, max_length=512, tokenizer=tokenizer)

    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    model = AutoModelForTokenClassification.from_pretrained("/project/rcc/hyadav/markuplm-base", num_labels=7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # print(model)

    label_list = ["B-" + x for x in list(id2label.values())]
    print(label_list)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    model.train()

    for epoch in range(10):  # loop over the dataset multiple times
        for batch in tqdm(dataloader):
            # get the inputs;
            inputs = {k:v.to(device) for k,v in batch.items()}

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(**inputs)
            logits = outputs.logits
            # labels = batch["labels"]

            # print(logits.size())
            # print(labels.size())
            # labels = torch.nn.functional.one_hot(labels, num_classes=7)

            # loss = criterion(logits.float().squeeze(), labels.squeeze())
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print("Loss:", loss.item())

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            preds, refs = get_labels(label_list, predictions, labels, device)
            metric = evaluate.load("seqeval")
            metric.add_batch( predictions=preds, references=refs,)

        eval_metric = compute_metrics()
        print(f"Epoch {epoch}:", eval_metric)
        print(f"Epoch {epoch}:", loss.item)




if __name__ == "__main__":
    main()