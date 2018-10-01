This topic modeling code is written using Python 3.6 in the Anaconda environment. The following libraries are required to run the program.


Note: Any Python environment running Python 3 should work however a number of dependencies are pre-installed in Anaconda and may need to be downloaded manually in any other installation.


I have used pip to install the following dependencies

1. pip install gensim
2. pip install json
3. pip install nltk
4. pip install spacy


I have used 3 files in my code:

1. The input json file 
The json file in the format provided

2. A .csv file converted from the json file
The .csv file consists of 2 columns:
	i. Column 1 contains document ID
	ii. Column 2 is the text for each document
 
3. Output .txt file
This is the output file with 5 topics for each of the input documents

