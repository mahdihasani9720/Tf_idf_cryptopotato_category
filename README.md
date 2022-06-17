# Tf_idf_cryptopotato_category
from tkinter import N
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np



while(True):
    # open cryptopotato_category.csv
    corpus  = pd.read_csv('cryptopotato_category.csv')

    # select part title in cryptopotato_category.csv
    docs = corpus ['title'].values


    # tf-idf
    vectorizer = TfidfVectorizer()
    tfidf_docs = vectorizer.fit_transform(docs)
    


    # query
    query = input('please inter your query for search( for exit please inter 0 ):\n')
    tfidf_query = vectorizer.transform([query])[0]

    # for stop while
    if( query == '0'):
        print('Exit')
        break

    # similarities
    cosines = []
    for d in tqdm(tfidf_docs):
        cosines.append(float(cosine_similarity(d, tfidf_query)))
    # Show tap 10 related documents    
    k = 10
    sorted_ids = np.argsort(cosines)
    for i in range(k):
        cur_id = sorted_ids[-i-1]
        print( cosines[cur_id], ' : ' ,docs[cur_id] )
        
