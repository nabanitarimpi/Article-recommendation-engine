import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import recommender_functions as rf


class RecommendationEngine():

   """
   A class to make article recommendations for a given user.


   Methods
   ---------
   read_clean_data(data_path, content_path):
      reads data from source and clean it as per requirement

   fit():
      creates the user-term matrix and similarity matrix and stores them as attributes of the class

   make_recommendations(user_id, self.df, self.article_content, num_rec):
      makes recommendations for a given user

   """

   def read_clean_data(self, data_path, content_path):

     """
     This method reads data from source and clean it as per requirement. It also stores the following as attributes :

     1. df : the dataframe containing information about user article interactions

     2. df_content : the dataframe with descriptions of articles 


     Parameters
     ------------
     data_path : str
          path to the datafile containing information about user article interaction

     content_path : str
          path to the datafile containing information about article content

     Returns
     ---------
     None

     """

     # read data from source files
     self.df = pd.read_csv(data_path)
     self.df_content = pd.read_csv(content_path)

     # clean the data 
     del self.df['Unnamed: 0']
     del self.df_content['Unnamed: 0']

     user_id = 1
     coded_dict = {}

     # convert email address of each user to a corresponding id in the dataframe 'df'
     for email in self.df['email'].unique():
        coded_dict[email] = user_id
        user_id += 1

     # create a new column in the dataframe that will hold the ids of users
     self.df['user_id'] = self.df['email'].apply(lambda x:coded_dict[x]) 

     # drop the original 'email' column as it is now encoded in user id
     self.df.drop('email', axis=1, inplace=True)

     # create a new column named 'interaction' in the dataframe 'df' which will return the value 1 for every user article interaction pair. If a user 
     # has interacted with the same article multiple times, for each interaction there will be one value (which is 1) in this column. 
     self.df['interaction'] = [1 for title in self.df['title']]

     # drop the 'doc_status' column from the dataframe 'df_content' as it has only one unique value
     self.df_content.drop('doc_status', axis=1, inplace=True)
     # drop the duplicate rows from 'df_content'
     self.df_content.drop_duplicates('article_id', inplace=True)



   def fit(self):

     """
     In this method, we perform the following three major steps :
      
     1. We create a user-item matrix from the dataframe 'df' whose rows are unique user ids, columns are unique article ids and entries are  
        number of interactions between users and articles.

     2. We take full names of all articles from the dataframe 'df_content' and convert it to a document-term matrix 'article_content' using the         
        CountVectorizer class of sklearn. The rows of this matrix are unique article names present in the dataset and the columns are tokens of the     
        vocabulary constructed from the same dataset. 

     3. We then take the dot product of this document-term matrix with itself to obtain a n_article x n_article (n_article being the total number of   
        unique articles present in the dataset) matrix with each cell representing how similar an article is to others.

     This method stores the following as attributes :

     1. user_item : the user-item matrix whose rows are unique user ids, columns are unique article ids and entries in each cell is the number
       of interactions between the corresponding user and article.

     2. article_content : a pandas dataframe containing the article id and its full name that we will use to extract out the content of the article.

     3. similarity_matrix : a matrix of dimension (n_article x n_article) where n_article is the number of unique articles present in the dataset. Each  
        cell in this matrix will represent how similar an article is to others. Note that in each row, the diagonal elements will be maximum as an    
        article is always most similar with itself.
     
     """

     # create the user-item matrix using the 'create_user_item_matrix' function
     self.user_item = rf.create_user_item_matrix(self.df)

     # select only the columns from 'df_content' that will be used in further analysis
     self.article_content = self.df_content[['doc_full_name', 'article_id']]

     # create an object of the class CountVectorizer
     countvec = CountVectorizer(analyzer=rf.text_to_word)
     # obtain the document-term matrix
     article_by_content = countvec.fit_transform(self.article_content['doc_full_name'])
     # take the dot product of the matrix with itself to find out how similar a document is to others 
     self.similarity_matrix = np.dot(article_by_content, article_by_content.transpose()).toarray()

     

   def make_recommendations(self, user_id, num_rec):

     """
     A method of the class that makes recommendations for a given user in the following way :

     1. If the input user is already present in our database, it performs user-based collaborative filtering to provide article recommendation to that 
        user.
     
     2. If we do not obtain desired number of recommendations (which is here specified by 'num_rec') from collaborative filtering, it then performs 
        content-based recommendation so that we can provide same number of recommendations for each user.

     3. If the input user is new, i.e., we do not have any interaction history of the user in our database, a recommendation list containing articles 
        with high popularity, i.e., large number of interactions (in a descending order) is provided.


     Parameters
     ------------
     user_id : int
         id of the input user for whom recommendations are to be made

     df : pandas dataframe
         dataframe defined from the file user-item-interactions.csv 

     article_content : pandas dataframe
         a dataframe containing ids and full names of articles

     num_rec : int
         the number of recommendations we want to make for the input user

     Returns
     ----------
     rec_ids : list
         a list of ids of the recommended articles
        
     rec_names : list
         a list of names of the recommended articles 

     """

     # if the user is present in our database
     if user_id in self.df['user_id'].unique():

        ### user based collaborative filtering ###
        # articles that our input user has interacted with 
        print("user based collaborative filtering")
        articles_read = rf.get_user_articles(user_id, self.df)[0]

        # users similar to our input user sorted according to the similarity score and interaction numbers
        neighbors_df = rf.get_top_sorted_users(user_id, self.df, self.user_item)
        similar_users = neighbors_df['neighbor_id'].values

        # list to store all the recommendations
        recs = []
 
        for user in similar_users:
            # articles that have no interaction with the input user but have interactions with similar users
            articles_not_read_1 = np.setdiff1d(rf.get_user_articles(user, self.df)[0], articles_read)
            
            # sort these articles according to the number of interactions in descending order
            articles_not_read_sorted_1 = rf.get_top_sorted_articles(articles_not_read_1, self.df)
            
            # now add these articles to the list 'recs' and consider only unique values
            recs.extend(articles_not_read_sorted_1)
            
            # if the length of the array exceeds the max. number of recommendations we want to make, break the loop  
            if len(set(recs)) > num_rec:
               break


        ### content based recommendation for users who still need recommendations ###
        if len(set(recs)) < num_rec:
            print("content based recommendation")
            # find out the articles that our user has interacted with and also have information about content
            articles_read_subset = np.intersect1d(articles_read, self.article_content['article_id'].unique()) 

            for article in articles_read_subset:
                # find out the articles similar in content
                similar_articles = list(rf.find_similar_articles(article, self.article_content, self.similarity_matrix)[2].keys())[1:]

                # choose articles that have similar content and also no interaction with the user
                articles_not_read_2 = np.setdiff1d(similar_articles, articles_read_subset, assume_unique=True)

                # sort the articles according to the popularity
                articles_not_read_sorted_2 = rf.get_top_sorted_articles(articles_not_read_2, self.df)

                # store them in the list
                recs.extend(articles_not_read_sorted_2)

                # if the length of the array exceeds the max. number of recommendations we want to make, break the loop  
                if len(set(recs)) > num_rec:
                   break 

        #  return the first 'num_rec' elements of our recommendation array in case it has more than num_rec entries   
        rec_ids = list(set(recs))[:num_rec]   

        # return the names of the recommended articles
        rec_names = []
        for rec_id in rec_ids:
            if rec_id in self.df['article_id'].unique():
               rec_names.extend(rf.get_article_names([rec_id], self.df))
            else:
               rec_names.extend(self.article_content[self.article_content['article_id'].isin([rec_id])]['doc_full_name'].values)
     
     # if the user is new
     else:
        ### rank based recommendation ###
        rec_ids, rec_names = rf.get_top_articles(num_rec, self.df)
        print("since the user is not in our database, we are giving the top articles as recommendations...")  
     
     return rec_ids, rec_names

def main():

     """
     The function to load and clean the data, build the recommendation engine and finally using it to make predictions.

     """

     if len(sys.argv) == 3:
        data_path, content_path = sys.argv[1:]

        # create an object of the class RecommendationEngine
        recommender = RecommendationEngine()

        print()
        print("loading data ...")
        print()
        recommender.read_clean_data(data_path, content_path)

        # fit the recommender
        recommender.fit() 
         
        # make recommendations
        print("article recommendation for user 10 :")
        print(recommender.make_recommendations(user_id=8, num_rec=10)[1])
        print()
        print(recommender.make_recommendations(user_id=6000, num_rec=10)[1])

     else:
        print('Please provide the filepath of the user article interaction '\
              'database as the first argument and the filepath of the '\
              'article content database as the second argument. \n\nExample: python '\
              'recommender.py data/user-item-interactions.csv data/articles_community.csv')


# this will only be executed when this module is run directly
if __name__ == "__main__":
     main() 
