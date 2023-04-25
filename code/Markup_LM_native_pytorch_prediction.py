import pandas as pd
import numpy as np
import pdb

# nodes_xpaths = pd.read_csv("nodes_xpaths_test.csv", delimiter="\t")
nodes_xpaths = pd.read_csv("../training_data/nodes_xpaths/Labeled_20150217144133_nodes_xpaths.csv",delimiter=",")

nodes_xpaths = nodes_xpaths.dropna()

nodes = nodes_xpaths['nodes'].values.tolist()
xpaths = nodes_xpaths['xpaths'].values.tolist()
node_labels = nodes_xpaths['Labels'].values.tolist()

# pdb.set_trace()

nodes = nodes_xpaths['nodes']
labels = nodes_xpaths['Labels']

df = nodes_xpaths[nodes_xpaths['Labels'] != 'NE']
df_pivoted = df.pivot(columns='Labels')['nodes']
names = df_pivoted['Name'].dropna().values.tolist()
address_list = df_pivoted['Address'].dropna().values.tolist()
beds = df_pivoted['Beds'].dropna().values.tolist()
contacts = df_pivoted['Contact'].dropna().values.tolist()
prices = df_pivoted['Price'].dropna().values.tolist()

i = 0
new_addresses = []
while(i < len(address_list)):
    address = address_list[i]
    num_words = len((str(address)).split())
    if num_words > 1:
        i = i + 1
    elif num_words <= 1 and i > 0:
        new_addresses.append(str(address_list[i-1]) + " " + str(address) +  " " + str(address_list[i+1]))
        i = i + 2

output = pd.DataFrame({'Name': names, 'Address': new_addresses, 'Price': prices, 'Beds': beds, 'Contact': contacts})
print(output.head(5))        





        









