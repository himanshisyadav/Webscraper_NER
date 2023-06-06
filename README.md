# Webscraper_NER

This repository contains the code and documentation for the Webscraping for Rental Property Data project with Prof. Milena Almagro at University of Chicago Booth School of Business in collaboration with Research Computing Center. 

## MarkupLM Language Model for Entity Classification

GitHub Repository for the pretrained language model: https://github.com/microsoft/unilm/tree/master/markuplm

### How to Run MarkupLM Pipeline on Midway3 Systems

1. Go to [code](code/) folder.
2. Download the model ```markuplm-base``` from: https://huggingface.co/microsoft/markuplm-base/tree/main and change the path for the model in the script ```Markup_LM_native_pytorch.py```.
3. If running through the ```sbatch``` script,
```
sbatch Markup_LM_native_pytorch.sbatch
```
else, use ```sinteractive``` to go to a compute node and follow the script ```sbatch``` script commands.




