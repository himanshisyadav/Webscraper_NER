# imports and load spacy english language package
import spacy
from spacy import displacy
from spacy import tokenizer


def print_entities(modeling,text):
    modeling.add_pipe('sentencizer')
    doc = modeling(text)
    #doc2 = nlp(text2)
    sentences = list(doc.sents)
    print(sentences)
    # # tokenization
    # for token in doc:
    #     print(token.text)
    # print entities
    ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
    print(ents)

 
#Load the text and process it
# I copied the text from python wiki
text_1 =("Python is an interpreted, high-level and general-purpose programming language. \
        Pythons design philosophy emphasizes code readability with its notable use of \
        significant indentation.Its language constructs and object-oriented approach aim \
        to help programmers write clear and logical code for small and large-scale projects")

text_2 = ("Chicago, IL")

nlp_1 = spacy.load('en_core_web_sm')
nlp_2 = spacy.load("models/output/model-best/") #load the model

print_entities(nlp_1,text_2)
print_entities(nlp_2,text_2)


