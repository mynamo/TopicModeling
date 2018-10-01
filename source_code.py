#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:15:13 2018

Copyright 2018 Aditi Kulkarni

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
#----------------------------------------------------------------------------------------------------
#Import the necessary libraries
#----------------------------------------------------------------------------------------------------

import gensim
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
import json
import pandas as pd
import csv
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()


nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

#----------------------------------------------------------------------------------------------------
#Function to parse document and extract tokens
#----------------------------------------------------------------------------------------------------

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

#----------------------------------------------------------------------------------------------------
#Functions to lemmatize the document i.e. group different forms of the same word
#----------------------------------------------------------------------------------------------------
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

#----------------------------------------------------------------------------------------------------
#Function that combines pre-processing methods for the text data
#----------------------------------------------------------------------------------------------------
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens
 
#----------------------------------------------------------------------------------------------------
#Files and directories
#----------------------------------------------------------------------------------------------------
my_dir = '/Users/aditikulkarni/Downloads/PracticeProjects/'
fileInput = my_dir+"topic_modeling_data.json"
fileOutput = my_dir+"topic_modeling_data.csv"
topics_modeled = my_dir+"topic_modeling.txt"

data = []
cnt = 0
with open(fileInput) as f:
    for line in f:
        data.append(json.loads(line))

outputFile = open(fileOutput, 'w') #load csv file
output = csv.writer(outputFile) #create a csv.write
output.writerow(data[0].keys())  # header row
for row in data:
    output.writerow(row.values()) #values row


#----------------------------------------------------------------------------------------------------
#Creating a corpus and applying the LDA model to each document
#----------------------------------------------------------------------------------------------------
with open(fileOutput) as f:
    for line in f:
        
        text_data = prepare_text_for_lda(line)
        if text_data:
            common_dictionary = corpora.Dictionary([text_data])
            common_corpus = [common_dictionary.doc2bow(text_data)]
            
            # Train the model on the corpus.
            ldamodel = gensim.models.ldamodel.LdaModel(common_corpus, num_topics = 5, id2word=common_dictionary)
            ldamodel.save('model3.gensim')
            topics = ldamodel.print_topics(num_topics=5, num_words=5)
            file=open(topics_modeled, "a+")
            
            for idx,doc in enumerate(common_corpus):
                if doc:
                    print('----------------------------------------------------------------------------------------------------')
                    vector = ldamodel[doc] 
                    print("ID:"+str(text_data[0])+"\n")
                    doc_top = sorted(ldamodel.get_document_topics(vector),key = lambda x: x[1])[-5:]
                    for d in doc_top:
                        print(topics[d[0]])
                    
                    file.write("ID:"+str(text_data[0])+"\n")
                    for d in doc_top:
                        file.write(topics[d[0]][1])
                        file.write("\n")
                    file.write("----------------------------------------------------------------------------------------------------\n")
                    

#----------------------------------------------------------------------------------------------------
#Output stored as a .txt file called topic_modeling.txt
#----------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------
#Further improvements: 
#This code can be made much more effective by updating the corpus and the dictionary at each update
#The PyLDAvis library can be used to visualize the text data as a cloud of relations between topics and words
#--------------------------------------------------------------------------------------------------------------------------------------
