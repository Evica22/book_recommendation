# -*- coding: utf-8 -*-
"""
Eva Slezakova 

DataSentics task  - Book recommendation model

"""

# import packages

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

df_all = pd.read_csv("df_all.csv")
######################################################################################################
# Collabrative Filtering Based Recommendation
######################################################################################################

####   with rating score
pivot_rating = df_all.pivot_table(index="ISBN", columns="User_ID", values="Book_Rating").fillna(0)
df_books=df_all.drop_duplicates(subset=['ISBN'], keep='first')


######################################################################################################
# Matrix Factorization (SVD)
######################################################################################################

def svd_recommendation(Tittle_ref='', Author_ref='', include=True, limit=10): # if include is True - model recommends also refrence book/books
    if ((Tittle_ref=='') | (Author_ref=='')):
        # quit the function and any function(s) that may have called it
        raise ValueError("Define either book's ISBN or it's Title and Author!")    
        
    # get ISBN ref 
    #if ISBN_ref=='':
    ISBN_ref = df_books.loc[(df_books['Book_Title'].str.contains(Tittle_ref, case=False, na=False)) & (df_books['Book_Author'].str.contains(Author_ref, case=False, na=False)), 'ISBN']#.iloc[0]
    if isinstance(ISBN_ref, str)==True:
        ISBN_ref=[ISBN_ref]

    #SVD = TruncatedSVD(n_components=12)
    SVD = TruncatedSVD(n_components=12, random_state=16)

    matrix= SVD.fit_transform(pivot_rating)
    matrix.shape

    corr=np.corrcoef(matrix) # Calculate the Pearson’s R correlation coefficient 
    corr.shape

    books_ISBN = pivot_rating.T.columns
    books_ISBN_list = list(books_ISBN)

    title=[]
    author=[]
    ISBN=[]
    cor=[]
    for var in ISBN_ref:
    #results
        index = books_ISBN_list.index(var)
        corr_index = corr[index]
        ISBN_suggestions = list(books_ISBN[(corr_index<1) & (corr_index>0.8)])
        for i in range(len(ISBN_suggestions)):
            temp = df_books.loc[(df_books['ISBN'] == ISBN_suggestions[i]) , ['ISBN','Book_Title', 'Book_Author']].iloc[0,:]
            ISBN.append(temp['ISBN'])
            title.append(temp['Book_Title'])
            author.append(temp['Book_Author'])
            cor.append(corr_index[(corr_index<1) & (corr_index>0.8)][i])
    
    final_sudgestion_df = pd.DataFrame({'Title' : title,
                           'Author' : author, 
                           'ISBN': ISBN,
                           'corr': cor}).sort_values('corr', ascending=False).drop_duplicates(subset=['ISBN'], keep='first')

    if include==False:
        final_sudgestion_df=final_sudgestion_df[~final_sudgestion_df['ISBN'].isin(ISBN_ref)]

    return final_sudgestion_df
    
    
#result = svd_recommendation(Tittle_ref = 'Lord of the rings', Author_ref = 'TOLKIEN')

# import pickle

# pickle.dump(svd_recommendation, open("svd.pkl","wb"))
