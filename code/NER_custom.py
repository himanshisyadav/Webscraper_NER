import spacy
import numpy as np
import pandas as pd

nlp = spacy.load("en_core_web_sm")
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
doc = nlp(text)
words = []
labels = []

for token in doc:
	words.append(token.text)
	labels.append('O') # As most of token will be non-entity (OUT). Replace this later with actual entity a/c the scheme.

df = pd.DataFrame({'word': words, 'label': labels})
# df.to_csv('ner-token-per-line.biluo', index=False) # biluo in extension to indicate the type of encoding, it is ok to keep csv

words  = df.word.values
ents = df.label.values
text = ' '.join(words)

from spacy.training import Example

doc = nlp.make_doc(text)
g = Example(doc, entities=ents)
X = [doc]
Y = [g]

print(X,Y)