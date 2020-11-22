import streamlit as st
import pandas as pd
import numpy as np
import os
import nltk
import re
import string
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import networkx as nx
nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}


def main():
    st.title("Extractive Summary")
    st.markdown("Are you tired of those long reports ??😔 ")
    st.markdown("Do you just want a short sweet summary?😃  ....I have a solution for you !!")
    st.markdown("You enter the report and the number of lines of summary you need ...!!")
    st.markdown("You will be provided  with the best summary !!😄")


    sentence  = st.text_area("Copy-Paste Here ...","Hi , Please enter the Text")
            # print(type(user_input))
            # print(user_input)

    num= st.number_input("Enter the number of Sentences in your summary:", 1, 30, step=1, key='n_estimators')
    if st.button("SUMMARY",key = "classify"):

            sentence  =  sent_tokenize(sentence)
            n  =  len(sentence)
            sentence = pd.Series(sentence)
            #Cleaning the data
            def msg_clean(msg):
                    msg = msg.lower()
                    tag = re.compile('<.*?>')
                    msg = re.sub(tag , ' ' , msg)
                    res = ' '.join([contraction_mapping.get(i, i) for i in msg.split()])
                    res = res.replace("'s" , "")
                    clean = [(char) for char in res if char not in string.punctuation]
                    clean = "".join(clean)
                    return clean

            sentence = sentence.apply(msg_clean)
            def preprocess(text):
                    result = []
                    for token in gensim.utils.simple_preprocess(text):
                            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) >2 and token not in stop_words:
                                result.append(token)
                    return result
            clean_data = sentence.apply(preprocess)
            clean_data = clean_data.apply(lambda x : " ".join(x))

            embeddings ={}
            with open("glove.6B.50d.txt", 'r', encoding="utf8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    embeddings[word] = vector
            sentence_vectors = []
            for i in clean_data:
                if len(i)!=0:
                    v = sum([embeddings.get(w, np.zeros((50,))) for w in i.split()])/ (len(i.split())+0.001)
                else :
                    v = np.zeros((50,))
                sentence_vectors.append(v)
            mat =  np.zeros([n,n])
            for i in range(len(sentence)):
                for j in range(len(sentence)):
                    if i!=j:
                        mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,50) , sentence_vectors[j].reshape(1,50))[0,0]

            nx_graph = nx.from_numpy_array(mat)
            scores = nx.pagerank(nx_graph)
            ranked_sentences =  sorted(((scores[i], s) for i,s in enumerate(sentence)), reverse = True)
            for i in range(num):
                st.write("{}) {}".format(i+1 ,ranked_sentences[i][1] ))






if __name__ == '__main__':
    main()
