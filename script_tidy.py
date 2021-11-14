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

users.describe()

users.loc[(users['Age']<13) | (users['Age']>100), 'Age']=np.nan # set users with suspicious age as nan
users['Age'] = users['Age'].fillna(users['Age'].mean())
plt.hist(users['Age'], bins=20 )
plt.title("Age of Users")
plt.show()

users[['City', 'State', 'Country']]= users['Location'].str.split(',', n=2, expand=True) # create new location variables

#################
# ratings
#################

ratings['Book_Rating'].describe()

plt.hist(ratings['Book_Rating'], bins=10 ) # histogram of rating scores
plt.title("Rating scores of Users")
plt.show()

ratings['User_ID'].value_counts().describe()   # number of ratings per user 
ratings = ratings.groupby('User_ID').filter(lambda x: len(x) >=30) # remove users with too few ratings 
ratings = ratings.groupby('User_ID').filter(lambda x: len(x) <4000) # remove users with way too many ratings 
ratings['User_ID'].value_counts().describe()

plt.hist(ratings['User_ID'].value_counts(), bins=80)  # histogram of num. of ratings per user
plt.title("Num. of ratings per user")
plt.show()

plt.hist(ratings.groupby('User_ID')['Book_Rating'].mean(), bins=50) # histogram of avg. rating per user
plt.title("Avg. rating per user")
plt.show() 

rating_aggr = pd.DataFrame(ratings.groupby('User_ID')['Book_Rating'].agg([np.size, np.mean]))
plt.scatter(x=rating_aggr['mean'],y=rating_aggr['size'], alpha=0.5)  # scatter plot of avg. rating of user per number of ratings 
plt.title("Ratings per user")
plt.xlabel("Avg. rating")
plt.ylabel("Num. of ratings")
plt.show()

#################
# books
#################

books.describe()

#   books[books['Year_Of_Publication'].astype(int)]  # Error -    incorrect data in publishing year
books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),:]

# correcting data 
temp = books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),:]

books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),'Book_Title'] = temp['Book_Title'].str.split('.\";', n=1, expand=True)[0]
books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),'Book_Author'] = temp['Book_Title'].str.split('.\";', n=1, expand=True)[1].str.replace('"', '')
books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),'Year_Of_Publication'] = temp['Book_Author']
books.loc[(books.Year_Of_Publication == 'DK Publishing Inc') | (books.Year_Of_Publication == 'Gallimard'),'Year_Of_Publication'] = temp['Publisher']

books.loc[(books.ISBN == '0789466953') | (books.ISBN == '078946697X') | (books.ISBN == '2070426769'),:]  # checking the result

books['Year_Of_Publication'] = pd.to_numeric(books['Year_Of_Publication']) 
books.loc[(books['Year_Of_Publication']<1950) | (books['Year_Of_Publication']>2010), 'Year_Of_Publication'] = np.nan # filter year of publishing
books['Year_Of_Publication'] = books['Year_Of_Publication'].fillna(books['Year_Of_Publication'].mean())

plt.hist(books['Year_Of_Publication'], bins=100) # histogram of publishing years
plt.title("Books' publishing years")
plt.show()

######################################################################################################
# joining data 
######################################################################################################

books_ratings = ratings.merge(books, left_on = 'ISBN', right_on = 'ISBN') # join books with their ratings
books_ratings.drop_duplicates(['User_ID','Book_Title', 'Book_Author'], inplace=True) # in case someone gave more than one rating

plt.hist(books_ratings['ISBN'].value_counts(), bins=20)  # histogram of num. of ratings per book
plt.title("Num. of ratings per book")
plt.show()

books_ratings = books_ratings.groupby('ISBN').filter(lambda x: len(x) >=50) # remove books with too few ratings 

books_rating_mean = pd.DataFrame(books_ratings.groupby('ISBN')['Book_Rating'].mean())
plt.hist(books_rating_mean['Book_Rating'], bins=20)  # histogram of avg.  rating per book
plt.title("Avg. rating per book")
plt.show()

books_aggr = pd.DataFrame(books_ratings.groupby('ISBN')['Book_Rating'].agg([np.size, np.mean]))
plt.scatter(x=books_aggr['mean'],y=books_aggr['size'], alpha=0.5)  # scatter plot of avg.  rating od a book per number of ratings 
plt.title("Books' ratings")
plt.xlabel("Avg. rating")
plt.ylabel("Num. of ratings")
plt.show()

df_all = books_ratings.merge(users, left_on = 'User_ID', right_on = 'User_ID') # join the users' information

missing_data = df_all.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")   # no missing data in the dataset
    
df_all.describe()

######################################################################################################
# Most popular books 
######################################################################################################

temp = books_aggr.sort_values(['mean'], ascending=False)  # sort aggregated books
popular_books = temp[temp['size']>200] # only books with enough reviews
popular_title=list(popular_books[:10].index)

title=[]
author=[]
for i in popular_title:
    title.append(df_all['Book_Title'].loc[df_all['ISBN']==i].values[0])
    author.append(df_all['Book_Author'].loc[df_all['ISBN']==i].values[0])

x=1
print('Top Books based on readers recommendations:')
for i in range(0, (len(title))):
    print(x, '. ', title[i], ' by ', author[i], sep=(""))
    x+=1
    
top_10 = pd.DataFrame({'Title' : title,
                       'Author' : author})
    
######################################################################################################
# Collabrative Filtering Based Recommendation
######################################################################################################

####   with rating score
pivot_rating = df_all.pivot_table(index="ISBN", columns="User_ID", values="Book_Rating").fillna(0)
#pivot_rating = df_all.pivot_table(index="ISBN", columns="User_ID", values="Liked").fillna(0)

df_books=df_all.drop_duplicates(subset=['ISBN'], keep='first')

######################################################################################################
# kNN recommendation model 
######################################################################################################


def nn_recommendation(Tittle_ref='', Author_ref='', ISBN_ref= '', include=True, limit=10): # if include is True - model recommends also refrence book/books
    if (ISBN_ref=='') & ((Tittle_ref=='') | (Author_ref=='')):
        # quit the function and any function(s) that may have called it
        raise ValueError("Define either book's ISBN or it's Title and Author!")    
        
    # get ISBN ref 
    if ISBN_ref=='':
        ISBN_ref = df_books.loc[(df_books['Book_Title'].str.contains(Tittle_ref, case=False, na=False)) & (df_books['Book_Author'].str.contains(Author_ref, case=False, na=False)), 'ISBN']#.iloc[0]
    if isinstance(ISBN_ref, str)==True:
        ISBN_ref=[ISBN_ref]
    
    # model
    book_sparse = csr_matrix(pivot_rating)
    model = NearestNeighbors(algorithm='brute')
    model.fit(book_sparse)
    
    title=[]
    author=[]
    ISBN=[]
    distance=[]
    for var in ISBN_ref:
        distances, suggestions = model.kneighbors(pivot_rating.loc[var, :].values.reshape(1, -1))
    #results
        ISBN_suggestions = pd.DataFrame(pivot_rating.index[suggestions])
        for i in ISBN_suggestions:
            temp = df_books.loc[(df_books['ISBN'] == ISBN_suggestions.iloc[0,i]) , ['ISBN','Book_Title', 'Book_Author']].iloc[0,:]
            ISBN.append(temp['ISBN'])
            title.append(temp['Book_Title'])
            author.append(temp['Book_Author'])
            distance.append(distances[0,i])
    
    final_sudgestion_df = pd.DataFrame({'Title' : title,
                           'Author' : author, 
                           'ISBN': ISBN,
                           'Distance': distance}).sort_values('Distance').drop_duplicates(subset=['ISBN'], keep='first')
    if include==False:  # if include == False, all books with the same title and author will be excluded from the recommendation
        final_sudgestion_df=final_sudgestion_df[~final_sudgestion_df['ISBN'].isin(ISBN_ref)]
    
    x=1
    print('Recommended books based on users preferences:')
    for i in range(min(len(final_sudgestion_df),limit)):
        print(x, '. ', final_sudgestion_df.iloc[i,0], ' by ', final_sudgestion_df.iloc[i,1], " (ISBN: ", final_sudgestion_df.iloc[i,2], ")", sep=(""))
        x+=1
    
    return final_sudgestion_df

final_sudgestion_nn = nn_recommendation(Tittle_ref = 'Lord of the rings', Author_ref = 'TOLKIEN', limit=10)

######################################################################################################
# Matrix Factorization (SVD)
######################################################################################################

def svd_recommendation(Tittle_ref='', Author_ref='', ISBN_ref= '', include=True, limit=10): # if include is True - model recommends also refrence book/books
    if (ISBN_ref=='') & ((Tittle_ref=='') | (Author_ref=='')):
        # quit the function and any function(s) that may have called it
        raise ValueError("Define either book's ISBN or it's Title and Author!")    
        
    # get ISBN ref 
    if ISBN_ref=='':
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

    if include==False:  # if include == False, all books with the same title and author will be excluded from the recommendation
        final_sudgestion_df=final_sudgestion_df[~final_sudgestion_df['ISBN'].isin(ISBN_ref)]
        
    x=1
    print('Top Books based on users preferences:')
    for i in range(min(len(final_sudgestion_df),limit)):
        print(x, '. ', final_sudgestion_df.iloc[i,0], ' by ', final_sudgestion_df.iloc[i,1], " (ISBN: ", final_sudgestion_df.iloc[i,2], ")", sep=(""))
        x+=1
    
    return final_sudgestion_df
    
final_sudgestion_svd = svd_recommendation(Tittle_ref = 'Lord of the rings', Author_ref = 'TOLKIEN')

