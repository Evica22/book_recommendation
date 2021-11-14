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

import os

path="C:/Users/cnb/Desktop/datasentics"
os.chdir(path)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

######################################################################################################
# load data 
######################################################################################################

users = pd.read_csv("data/BX-Users.csv",sep=';', encoding="latin-1", error_bad_lines=False, warn_bad_lines=False)
users.columns = users.columns.str.replace("-", "_")  # remove all "-" from column names

ratings = pd.read_csv("data/BX-Book-Ratings.csv",sep=';', encoding="latin-1", error_bad_lines=False, warn_bad_lines=False)
ratings.columns = ratings.columns.str.replace("-", "_")  # remove all "-" from column names

books = pd.read_csv("data/BX-Books.csv",sep=';', encoding="latin-1", error_bad_lines=False, warn_bad_lines=False)
books = books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)
books.columns = books.columns.str.replace("-", "_")  # remove all "-" from column names

######################################################################################################
# clean data 
######################################################################################################

#################
# users 
#################

users.loc[(users['Age']<13) | (users['Age']>100), 'Age']=np.nan # set users with suspicious age as nan
users['Age'] = users['Age'].fillna(users['Age'].mean())

#################
# ratings
#################

ratings = ratings.groupby('User_ID').filter(lambda x: len(x) >=30) # remove users with too few ratings 
ratings = ratings.groupby('User_ID').filter(lambda x: len(x) <4000) # remove users with way too many ratings 

#################
# books
#################

#   books[books['Year_Of_Publication'].astype(int)]  # Error -    incorrect data in publishing year
books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),:]

# correcting data 
temp = books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),:]

books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),'Book_Title'] = temp['Book_Title'].str.split('.\";', n=1, expand=True)[0]
books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),'Book_Author'] = temp['Book_Title'].str.split('.\";', n=1, expand=True)[1].str.replace('"', '')
books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),'Year_Of_Publication'] = temp['Book_Author']
books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),'Year_Of_Publication'] = temp['Publisher']

books['Year_Of_Publication'] = pd.to_numeric(books['Year_Of_Publication']) 
books.loc[(books['Year_Of_Publication']<1950) | (books['Year_Of_Publication']>2020), 'Year_Of_Publication'] = np.nan # filter year of publishing
books['Year_Of_Publication'] = books['Year_Of_Publication'].fillna(books['Year_Of_Publication'].mean())


######################################################################################################
# joining data 
######################################################################################################

books_ratings = ratings.merge(books, left_on = 'ISBN', right_on = 'ISBN') # join books with their ratings
books_ratings.drop_duplicates(['User_ID','Book_Title', 'Book_Author'], inplace=True) # in case someone gave more than one rating

books_ratings = books_ratings.groupby('ISBN').filter(lambda x: len(x) >=50) # remove books with too few ratings 

books_aggr = pd.DataFrame(books_ratings.groupby('ISBN')['Book_Rating'].agg([np.size, np.mean]))

df_all = books_ratings.merge(users, left_on = 'User_ID', right_on = 'User_ID') # join the users' information


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
