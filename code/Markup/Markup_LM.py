from transformers import AutoProcessor, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import evaluate
import pdb

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class HTMLDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

processor = AutoProcessor.from_pretrained("/project/rcc/hyadav/markuplm-base")
processor.parse_html = False
model = AutoModelForTokenClassification.from_pretrained("/project/rcc/hyadav/markuplm-base", num_labels=2)

# nodes = ["hello", "world"]
# xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"]
# node_labels = [1, 2]

nodes_xpaths = pd.read_csv("nodes_xpaths_test.csv", delimiter="\t")
nodes_xpaths = nodes_xpaths.replace(np.nan, "", regex=True)

nodes = nodes_xpaths['nodes'].values.tolist()
xpaths = nodes_xpaths['xpaths'].values.tolist()
node_labels = [0] * len(nodes_xpaths)

# pdb.set_trace()

train_split_cent, val_split_cent, test_split_cent = 0.8, 0.1, 0.1
indices = list(range(len(nodes)))
train_indices = indices[: int(np.floor(train_split_cent * len(nodes)))]
val_indices = indices[ int(np.floor(train_split_cent * len(nodes))) : int(np.floor((train_split_cent + val_split_cent) * len(nodes)))]
test_indices = indices [ int(np.floor((train_split_cent + val_split_cent) * len(nodes))) :]

train_encodings = processor(nodes=nodes[: int(np.floor(train_split_cent * len(nodes)))] , xpaths=xpaths[: int(np.floor(train_split_cent * len(nodes)))], node_labels=node_labels[: int(np.floor(train_split_cent * len(nodes)))], return_tensors="pt",  truncation=True, max_length=512)
val_encodings = processor(nodes=nodes[ int(np.floor(train_split_cent * len(nodes))) : int(np.floor((train_split_cent + val_split_cent) * len(nodes)))], xpaths=xpaths[ int(np.floor(train_split_cent * len(nodes))) : int(np.floor((train_split_cent + val_split_cent) * len(nodes)))], node_labels=node_labels[ int(np.floor(train_split_cent * len(nodes))) : int(np.floor((train_split_cent + val_split_cent) * len(nodes)))], return_tensors="pt",  truncation=True, max_length=512)
test_encodings= processor(nodes=nodes[ int(np.floor((train_split_cent + val_split_cent) * len(nodes))) :], xpaths=xpaths[ int(np.floor((train_split_cent + val_split_cent) * len(nodes))) :], node_labels=node_labels[ int(np.floor((train_split_cent + val_split_cent) * len(nodes))) :], return_tensors="pt",  truncation=True, max_length=512)

train_labels = node_labels[: int(np.floor(train_split_cent * len(nodes)))]
val_labels = node_labels[ int(np.floor(train_split_cent * len(nodes))) : int(np.floor((train_split_cent + val_split_cent) * len(nodes)))]
test_labels = node_labels[ int(np.floor((train_split_cent + val_split_cent) * len(nodes))) :]

train_dataset = HTMLDataset(train_encodings, train_labels)
val_dataset = HTMLDataset(val_encodings, val_labels)
test_dataset = HTMLDataset(test_encodings, test_labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# pdb.set_trace()

metric = evaluate.load("accuracy")

trainer = Trainer(model=model(train_encodings),args=training_args,train_dataset=train_dataset,eval_dataset=val_dataset,compute_metrics=compute_metrics)

trainer.train()


# with torch.no_grad():
#     outputs = model(**encoding)

# loss = outputs.loss
# logits = outputs.logits