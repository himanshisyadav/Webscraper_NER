## MarkupLM Language Model for Entity Classification

GitHub Repository for the pretrained language model: https://github.com/microsoft/unilm/tree/master/markuplm

### How to Run MarkupLM Pipeline on Midway3 Systems

1. Access the code from [this](./) folder.
2. Download the model ```markuplm-base``` from: https://huggingface.co/microsoft/markuplm-base/tree/main and change the path for the model in the script ```Markup_LM_native_pytorch.py```.
3. If running through the ```sbatch``` script,
```
sbatch Markup_LM_native_pytorch.sbatch
```
else, use ```sinteractive``` to go to a compute node and follow the script ```sbatch``` script commands.
