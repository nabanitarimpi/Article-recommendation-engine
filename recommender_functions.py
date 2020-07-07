# import necessary libraries

import pandas as pd
import numpy as np


import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])


# define the functions that will be required to make recommendations later

def create_user_item_matrix(df):
    
    """
    This function returns a matrix which holds the information about interactions between users and articles.
    
    Parameter
    ----------
    df : pandas dataframe 
       dataframe as defined at the top of the notebook from the file user-item-interactions.csv
    
    Returns
    ---------
    user_item : matrix
       a matrix whose rows are unique user ids, columns are unique article ids and entries in each cell is the number
       of interactions between the corresponding user and article. We fill all the null values with 0 to 
       denote no interaction between the corresponding users and articles. 
  
    """
    
    user_item = df.groupby(['user_id', 'article_id'])['interaction'].min().unstack().fillna(0)
    
    return user_item 


def get_top_sorted_users(user_id, df, user_item):
    
    """
    This function returns a dataframe which contains all the neighbors of the input user sorted according to their
    similarity scores first and then by the total number of interactions.
    
    Parameters
    ------------
    user_id : int
         an input user id
        
    df : pandas dataframe
         dataframe as defined at the top of the notebook from the file user-item-interactions.csv
    
    user_item : (pandas dataframe) matrix 
         a users by articles matrix where non-zero entries represents that a user has interacted with an article
         and 0 stands for no interaction.
    
    Returns
    ---------
    neighbors_df : pandas dataframe
         a dataframe with the following columns:
         1. neighbor_id : a neighbor user_id
         2. similarity_score : measure of the similarity between the input user and its neighbors
         3. total_interactions : the number of articles viewed by a neighbor user
   
    """
    most_similar_users, similarity_score = [], []
    
    # loop through all other users    
    for other_id in range(len(user_item)):
        if other_id != user_id-1: # since the indexing start from zero in python
            # store the similarity score for every other user
            similarity_score.append(np.dot(user_item.iloc[user_id-1, :], user_item.iloc[other_id, :]))
            # store the id of every other user
            most_similar_users.append(other_id+1)

    # store the total number interactions for each similar user
    total_interactions = [df.groupby('user_id').count()['interaction'].values[id-1] for id in most_similar_users]
    
    # construct the dataframe
    neighbors_df = pd.DataFrame([most_similar_users, similarity_score, total_interactions]).transpose() 
    neighbors_df.columns = ['neighbor_id', 'similarity_score', 'total_interactions']
    # sort first by similarity score and then by number of interactions
    neighbors_df.sort_values(by=['similarity_score','total_interactions'], ascending=False, inplace=True)
    
    return neighbors_df 


def get_article_names(article_ids, df):
    
    """
    This function, given a list of article ids, returns a corresponding list containing the article titles.
    
    Parameters
    ------------
    article_ids : list
        a list of article ids
        
    df : pandas dataframe
        dataframe as defined at the top of the notebook from the file user-item-interactions.csv
    
    Returns
    ---------
    article_names : list
          a list of article names associated with the input list of article ids 
                    
    """
    
    article_names = df[df['article_id'].isin(article_ids)]['title'].unique()
    
    return list(article_names) 


def get_user_articles(user_id, df):
    
    """
    This function provides a list of article ids and corresponding titles that a user has interacted with.
    
    Parameters
    ------------
    user_id : int
        an input user id
        
    df : pandas dataframe
        dataframe as defined at the top of the notebook from the file user-item-interactions.csv
    
    Returns
    ---------
    article_ids : list
          a list of the article ids that the user has interacted with
          
    article_names : list
          a list of article names associated with the list article_ids 
    
    """
    
    article_ids = df.query('user_id==@user_id')['article_id'].unique()
    article_names = df[df['article_id'].isin(article_ids)]['title'].unique()
    
    return article_ids, article_names 


def get_top_sorted_articles(article_ids, df):
    
    """
    Given a list of article ids, this function sorts them according to the number of interactions they have with
    users in a descending order.
    
    Parameters
    ------------
    article_ids : list
        a list of article ids
        
    df : pandas dataframe
        dataframe as defined at the top of the notebook from the file user-item-interactions.csv
    
    Returns
    ---------
    sorted_article_ids : array
         an array of article ids sorted according to the number of interactions
    
    """
    
    df_new = df.groupby('article_id').count().reset_index()[['article_id', 'interaction']].sort_values('interaction', ascending=False)
    sorted_article_ids = df_new[df_new['article_id'].isin(article_ids)]['article_id'].values
    
    return sorted_article_ids


def text_to_word(text):
    
    """
    A function to clean an input text. The steps followed from the text cleaning are :
     
     1. Normalization i.e. conversion to lower case and punctuation removal
     2. Tokenization 
     3. Stop words removal
     4. Lemmatization
    
    Parameter 
    -----------
      text : str 
        the input text to be cleaned
      
    Returns 
    ----------
      lemm_token_list : list
             a list of tokens obtained after cleaning the text
        
    """    
    
    token_list = word_tokenize(re.sub(r"[^a-z0-9]", " ", text.lower()))
    token_nostop_list = [token for token in token_list if token not in stopwords.words("english")]
    pos_dict = {"N":wordnet.NOUN, "J":wordnet.ADJ, "V":wordnet.VERB, "R":wordnet.ADV}
    
    lemm_token_list = set()
    for token,pos in nltk.pos_tag(token_nostop_list):
        try:
            lemm_token_list.add(WordNetLemmatizer().lemmatize(token, pos_dict[pos[0]]))
        except:
            pass
        
    return list(lemm_token_list)    


def find_similar_articles(article_id, article_content, similarity_matrix):
    
    """
    Given an article id, this function returns a list of articles which are similar to that of the input article
    in terms of their content.
    
    Parameters
    ------------
    article_id : str
        id of the input article
        
    article_content : pandas dataframe
        a dataframe containing ids and full names of articles
        
    similarity_matrix : array
        an (n_article x n_article) numpy array where n_article is the number of unique articles present in the 
        dataset and each entry in this array will represent how similar an article is to others 
        
        
    Returns
    ---------
    similar_id : list
         ids of the users similar to the input user
         
    similarity_score : list    
         similarity scores of the neighbor users
         
    similar_dict : dict
         a dictionary whose keys are ids of similar users and values are the corresponding similariy scores sorted
         according to their similarity scores
    
    """
    
    # find out which row of the dataframe does the input article id belong to
    article_row = np.where(article_content['article_id']==article_id)[0][0]

    # find out the row numbers of similar articles
    similar_row = list(np.where(similarity_matrix[article_row] > 2)[0])

    # store the corresponding similarity scores
    similarity_score = list(similarity_matrix[article_row, similar_row])
    
    # store the ids of similar articles
    similar_id = list(article_content.iloc[similar_row]['article_id'])
    
    similar_dict = {}
    for similar_id,score in zip(similar_id, similarity_score):
        similar_dict[similar_id] = score
        
    # sort the dictionary according to the similarity scores
    similar_dict = {k:v for k,v in sorted(similar_dict.items(), key=lambda x:x[1], reverse=True)}    
    
    return similar_id, similarity_score, similar_dict


def get_top_articles(n, df):
    
    """
    This function determines the most popular articles based on the number of interactions and returns
    the corresponding ids and names.
    
    Parameters
    ------------
    n : int
     The number of top articles to return
     
    df : pandas dataframe
      dataframe defined from the file user-item-interactions.csv 
    
    Returns
    -----------
    top_articles_id : list
       A list of the top 'n' article ids
       
    top_articles : list 
       A list of the top 'n' article names    
    
    """
    
    top_articles_id = list(df.groupby('article_id').count()['user_id'].sort_values(ascending=False).index[:n])
    top_articles = list(df[df['article_id'].isin(top_articles_id)]['title'].unique())
    
    return top_articles_id, top_articles 
