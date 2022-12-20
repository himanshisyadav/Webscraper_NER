import spacy

nlp=spacy.load('bert-base-NER/config.json')

# Getting the pipeline component
ner=nlp.get_pipe("ner")