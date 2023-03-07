from transformers import AutoProcessor, AutoModelForTokenClassification, get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
import evaluate
import pdb

from tqdm.auto import tqdm

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class HTMLDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        for key, val in self.encodings.items():
            val = torch.squeeze(val)
            pdb.set_trace()
            item = {key: val[idx]}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)


# nodes = ["hello", "world"]
# xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"]
# node_labels = [1, 2]

nodes_xpaths = pd.read_csv("nodes_xpaths_test.csv", delimiter="\t")
nodes_xpaths = nodes_xpaths.replace(np.nan, "", regex=True)

nodes = nodes_xpaths['nodes'].values.tolist()
xpaths = nodes_xpaths['xpaths'].values.tolist()
node_labels = [0] * len(nodes_xpaths)

train_split_cent, val_split_cent, test_split_cent = 0.8, 0.1, 0.1
indices = list(range(len(nodes)))
train_indices = indices[: int(np.floor(train_split_cent * len(nodes)))]
val_indices = indices[ int(np.floor(train_split_cent * len(nodes))) + 1: int(np.floor((train_split_cent + val_split_cent) * len(nodes)))]
test_indices = indices [ int(np.floor((train_split_cent + val_split_cent) * len(nodes))) :]

processor = AutoProcessor.from_pretrained("/project/rcc/hyadav/markuplm-base")
processor.parse_html = False

# print(len(nodes[ : int(np.floor(train_split_cent * len(nodes)))]))
# print(len(nodes[ int(np.floor(train_split_cent * len(nodes))) + 1 : int(np.floor((train_split_cent + val_split_cent) * len(nodes)))]))
# print(len(nodes[ int(np.floor((train_split_cent + val_split_cent) * len(nodes))) + 1 :]))

train_encodings = processor(nodes=nodes[ : int(np.floor(train_split_cent * len(nodes)))] , xpaths=xpaths[: int(np.floor(train_split_cent * len(nodes)))], node_labels=node_labels[: int(np.floor(train_split_cent * len(nodes)))], return_tensors="pt",  truncation=True, max_length=512, padding=True)
val_encodings = processor(nodes=nodes[ int(np.floor(train_split_cent * len(nodes))) + 1 : int(np.floor((train_split_cent + val_split_cent) * len(nodes)))], xpaths=xpaths[ int(np.floor(train_split_cent * len(nodes))) + 1 : int(np.floor((train_split_cent + val_split_cent) * len(nodes)))], node_labels=node_labels[ int(np.floor(train_split_cent * len(nodes))) + 1 : int(np.floor((train_split_cent + val_split_cent) * len(nodes)))], return_tensors="pt",  truncation=True, max_length=512, padding=True)
test_encodings= processor(nodes=nodes[ int(np.floor((train_split_cent + val_split_cent) * len(nodes))) + 1 :], xpaths=xpaths[ int(np.floor((train_split_cent + val_split_cent) * len(nodes)))  + 1:], node_labels=node_labels[ int(np.floor((train_split_cent + val_split_cent) * len(nodes))) + 1:], return_tensors="pt",  truncation=True, max_length=512, padding=True)

train_labels = node_labels[: int(np.floor(train_split_cent * len(nodes)))]
val_labels = node_labels[ int(np.floor(train_split_cent * len(nodes)))  + 1: int(np.floor((train_split_cent + val_split_cent) * len(nodes)))]
test_labels = node_labels[ int(np.floor((train_split_cent + val_split_cent) * len(nodes))) + 1:]

# train_dataset = HTMLDataset(train_encodings, train_labels)
# val_dataset = HTMLDataset(val_encodings, val_labels)
# test_dataset = HTMLDataset(test_encodings, test_labels)

# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
# val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=8)
# test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=8)

model = AutoModelForTokenClassification.from_pretrained("/project/rcc/hyadav/markuplm-base", num_labels=1)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_labels)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
                            )

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    train_encodings.to(device)
    outputs = model(**train_encodings)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)



metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()




