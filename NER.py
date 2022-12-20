
#!/usr/bin/env python
# coding: utf-8

# Import Packages (Same as before)

###################### STEP 1 ###################### 
#import packages
import time
import os
import pandas as pd
import numpy as np
import random
import re
import pdb

# !pip3 install -U spacy
import spacy

from time import sleep
from os import path
from pandas import DataFrame
from random import randint
import spacy
from spacy import displacy


# Function to get all the contents from a particular webpage


def get_contents(soup, content_text):
  try:
    parents_blacklist=['[document]','html','head',
                       'style','script','body',
                       'section','tr',
                       'td','label','ul','header',
                       'aside']
    content=''
    text=soup.find_all(text=True)

    
    for t in text:
        if t.parent.name not in parents_blacklist and len(t) > 5:
            content=content+t+' '
    content_text.append(content)
  except Exception:
    content_text.append('')
    pass



# Function to get phone numbers, address and prices using regular expressions

# def get_regex_data(webpage):
#     nlp = spacy.load('/home/hyadav/.conda/envs/spacy_venv/lib/python3.10/site-packages/en_core_web_sm/en_core_web_sm-3.3.0')
#     page=requests.get(webpage)
#     soup=BeautifulSoup(page.text,'html.parser')
#     content_text = []
#     get_contents(soup, content_text)
#     contacts = re.findall(r'\(?[0-9]{3}\)?\/?\.?-?\s?[0-9]{3}-?\.?-?\s?[0-9]{4}', content_text[0])
#     address = regex.findall(r'[\s\n]*[1-9][0-9]{,4} (?:[A-Z][a-zA-Z]*,?\s*){,5} Chicago,?[\w\s,]{,3}\s*[0-9]{,6}?', content_text[0])
#     address = [addi.strip() for addi in address]
#     prices = []
#     prices = re.findall(r'\$[1-9]?[0-9]{,3},?[0-9]{,3}\s*-\s*\$?[1-9]?[0-9]{,3},?[0-9]{,3}', content_text[0])
#     print("\nContacts are:", contacts)
#     print("\nAddresses are:", address)
#     print("\nPrices are:" , prices)


# page_2015 ="http://web.archive.org/web/20150217144133/https://www.apartments.com/chicago-il/"
# page_2022 = 'http://web.archive.org/web/20220819053652/https://www.apartments.com/chicago-il/'


# import en_core_web_sm
# nlp = en_core_web_sm.load()
# print(nlp.pipe_names)
# ner = nlp.get_pipe("ner")


info_2015 = pd.read_csv("Info_2015.csv", sep = '\t')
info_2022 = pd.read_csv("Info_2022.csv", sep = '\t')
info_craig = pd.read_csv("apartments_com_prices_2015_2022.csv")
# pdb.set_trace()
info_2015 = info_2015.drop(columns = ['amenities', 'snapshot_link', 'property_link', 'address', 'price_range', 'bed_range', 'contact', 'name'], axis = 1)
info_2022 = info_2022.drop(columns = ['amenities', 'snapshot_link', 'property_link', 'address', 'price_range', 'bed_range', 'contact', 'name'], axis = 1)

def convert_to_ner_data(df):
    data = list()
    for col in df.columns:
        if col != 'Unnamed: 0':
            data = data + df[col].apply(lambda x: (x, {"entities": [(0, len(str(x)), col)]})).to_list()
    return data


training_data = convert_to_ner_data(info_2015)
test_data = convert_to_ner_data(info_2022)
craig_data = convert_to_ner_data(info_craig)

from spacy.tokens import DocBin
from tqdm import tqdm

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification

# nlp = spacy.blank("en") # load a new spacy model
# nlp.to_disk("en_himi")
# nlp_from_disk = spacy.load("himi")
# nlp = nlp_from_disk

# tokenizer = AutoTokenizer.from_pretrained("bert-base-NER")
# model = AutoModelForTokenClassification.from_pretrained("bert-base-NER")
# nlp = pipeline("ner", model=model, tokenizer=tokenizer)


nlp = spacy.blank("en")
db = DocBin()
for text, annot in tqdm(training_data): # data in previous format
    doc = nlp.make_doc(str(text)) # create doc object from text
    ents = []
    for start, end, label in annot['entities']: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode='contract')
        if span is None:
            print('Skipping entity')
        else:
            ents.append(span)
    try:
        doc.ents = ents # label the text with the ents
        db.add(doc)
    except:
        print(text, annot)

db.to_disk('./train.spacy') # save the docbin object
db.to_disk('./valid.spacy')



