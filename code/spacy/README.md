## How to Run Spacy Pipeline

1. Clone the repository on your system

```
git clone git@github.com:himanshisyadav/Webscraper_NER.git
```

2. Create a ```conda``` environment using the ```requirements.txt``` file in the [code](../code/) folder.
```
cd code/
conda create --name <env> --file requirements.txt
```
```
python3 NER_training.py
```

3. Obtain base_config.cfg from https://spacy.io/usage/training 
4. Run training code

```
python3 -m spacy init fill-config base_config.cfg data/config.cfg
```
```
python3 -m spacy train data/config.cfg --paths.train ./train.spacy --paths.dev ./valid.spacy --output models
```
5. To update the ```requirements.txt``` file in [code](code/) folder, run

```
conda list --explicit > requirements.txt
```

## How to Visualize with Jupyter Notebook

1. To open a notebook, run

```
jupyter notebook
```
2. Select a ```.ipynb``` file to open, edit and visualize test results.
