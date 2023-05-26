import requests
import time
import os
import pandas as pd
import numpy as np
import random
import re

import spacy
from spacy import displacy
from time import sleep
from os import path
from pandas import DataFrame
from bs4 import BeautifulSoup
from random import randint
from pathlib import Path

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

def preprocessor_final(text):
    if isinstance((text), (str)):
        text = re.sub('<[^>]*>', ' ', text)
#         text = re.sub('[\W_^\$]+', ' ', text.lower())
        return text
    if isinstance((text), (list)):
        return_list = []
        for i in range(len(text)):
#             temp_text = re.sub('<[^>]*>', ' ', text[i])
#             text = re.sub('[\W_^\$]+', ' ', text.lower())
            return_list.append(temp_text)
        return(return_list)
    else:
        pass

def display_entities(model_path, content):
    nlp = spacy.load(model_path)
    doc = nlp(content)
    svg = displacy.render(doc, style="ent", jupyter = True)
    print(svg)
    output_path = Path("./plot.svg") 
    output_path.open("w", encoding="utf-8").write(svg)

def main():
    page_2015 = "https://web.archive.org/web/20150217144133/https://www.apartments.com/chicago-il/"
    page_2022 = 'https://web.archive.org/web/20220819053652/https://www.apartments.com/chicago-il/'

    page=requests.get(page_2015)
    soup=BeautifulSoup(page.text,'html.parser')
    content_text = []
    get_contents(soup, content_text)
    # print(content_text)
    content = "".join(content_text)

    clean_content = preprocessor_final(content)
    clean_content = ' '.join(clean_content.split())
    
    print(clean_content)

    display_entities("en_core_web_sm", clean_content)


if __name__ == "__main__":
    main()