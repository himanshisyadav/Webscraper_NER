import spacy
import pandas as pd 

nlp = spacy.load("./models/output/model-best")

def convert_to_ner_data(df):
    data = list()
    for col in df.columns:
        if col != 'Unnamed: 0':
            data = data + df[col].apply(lambda x: (x, {"entities": [(0, len(str(x)), col)]})).to_list()
    return data


info_2022 = pd.read_csv("Info_2022.csv", sep = '\t')
info_2022 = info_2022.drop(columns = 'amenities', axis = 1)
# test_data = convert_to_ner_data(info_2022)

# print(info_2022.head())
# print(test_data)

# print(type(test_data))


for col in info_2022.columns:
	print(col)