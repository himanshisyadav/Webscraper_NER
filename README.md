# Webscraper_NER

This repository contains the code and documentation for the Webscraping for Rental Property Data project with Prof. Milena Almagro at University of Chicago Booth School of Business in collaboration with Research Computing Center. 

## How to Run

Create a ```conda``` environment using the ```requirements.txt``` file in the [code](code/) folder.
```
conda create --name <env> --file requirements.txt
```
```
python3 NER_training.py
```
Obtain base_config.cfg from https://spacy.io/usage/training 

```
python -m spacy init fill-config base_config.cfg data/config.cfg
```
```
python3 -m spacy train data/config.cfg --paths.train ./train.spacy --paths.dev ./valid.spacy --output models
```
