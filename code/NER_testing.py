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

def convert_to_ner_data_craig(df):
    data = list()
    simple_data = list()
    for col in df.columns:
    	if col == 'Address': 
    		new_name = 'address'
    	elif col == "Beds":
    		new_name = 'bed_range'
    	elif col == "Rent":
    		new_name = 'price_range'
    	else:
    		new_name = col
    	if col != 'Unnamed: 0' and col != 'Link':
    		simple_data = simple_data + df[col].apply(lambda x: (x, new_name)).to_list()
    return simple_data

def convert_to_ner_data_thiya(df):
    data = list()
    simple_data = list()
    for col in df.columns:
    	if col == 'Company': 
    		new_name = 'owner'
    	else:
    		new_name = col
    	if col != 'Unnamed: 0' and col != 'Name':
    		simple_data = simple_data + df[col].apply(lambda x: (x, new_name)).to_list()
    return simple_data



if __name__ == "__main__":
	info_craig = pd.read_csv("apartments_com_prices_2015_2022.csv")
	info_thiya = pd.read_csv("pm.csv")

	test_data_craig_simple = convert_to_ner_data_craig(info_craig)
	test_data_thiya_simple = convert_to_ner_data_thiya(info_thiya)

	nlp = spacy.load("models/model-best")

	# pdb.set_trace()

	correct = 0
	total = 0

	for element in test_data_thiya_simple:
		if isinstance(element[0], str):
			text = " ".join(element[0].split())
		else:
			continue
		doc = nlp(text)
		for word in doc.ents:
			if word.label_ == element[1]:
				total = total + 1
				correct = correct + 1
				# print("Correct", total, correct, word.text, word.label_, element[1])
			else:
				total = total + 1
				print("Incorrect", total, correct, word.text, word.label_, element[1])

	print("Corret, Total, Accuracy: ", correct, total, (correct/total) * 100)

