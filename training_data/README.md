# Data Annotation Process

Steps:

1. Raw data is present in Webscraper_NER/training_data/nodes_xpaths directory
2. Make a copy of the .csv files.
3. Go to Excel
4. Import Data as follows: On the Data tab, in the Get & Transform Data group, click From Text/CSV. In the Import Data dialog box, locate and double-click the text file that you want to import, and click Import. Then click on Transform data and pick the "Use first row as headers" from the top right corner, close and load. 
5. Then add a column to the database for labeling with the following labels:
    * Name
    * Owner
    * Address
    * Price
    * Beds
    * Amenities
    * Contact
6. Consult the data from info_2015.csv and info_2022.csv to correctly label this new dataset
7. Division of files:
   * Himi: 20150217144133_nodes_xpaths.csv (finished), 20220403132436_nodes_xpaths.csv (finished)
   * Katy: 20220121035604_nodes_xpaths.csv (finished)
            20220321160908_nodes_xpaths.csv (finished)
   * Stancy:
   * Fern: 
