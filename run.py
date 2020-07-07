# import packages from flask library
from flask import Flask
from flask import render_template, request

import pandas as pd
import recommender_functions as rf
import recommender as r
import joblib

#nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

app = Flask(__name__)

# load data
df = joblib.load("df.pkl")
df_content = joblib.load("df_content.pkl")
similarity_mat = joblib.load("similarity_mat.pkl")

article_content = df_content[['doc_full_name', 'article_id']]

# index webpage displays data visuals and receives user input text for model
@app.route('/')
@app.route('/index')


def index():
    
    """
    A function to render the homepage and index webpage
    
    """
    
    top_ids = rf.get_top_articles(50, df)[0]
    
    select_top_headings = list(article_content[article_content['article_id'].isin(top_ids)]['doc_full_name'].values)
    
    return render_template(
                           'homepg.html',
                           select_top_headings=select_top_headings
                          ) 

 
# web page that handles user query and displays recommendations
@app.route('/go')

def go():
    
    """
    A function to render the query web page
    
    """
    # save user input in query
    query = request.args.get('query', '')

    # description of the query
    query_art_body = df_content[df_content['doc_full_name']==query]['doc_body'].values[0]
    query_art_body = query_art_body.replace("\\r\\n", "")
    
    # find out the corresponding article id from 'article_content'
    query_id = article_content[article_content['doc_full_name']==query]['article_id'].values[0]

    # find out the similar articles
    similar_arts_dict = rf.find_similar_articles(query_id, article_content, similarity_mat)[2]

    # make content based recommendations
    rec = []
 
    for k,v in similar_arts_dict.items():
        rec.extend(article_content[article_content['article_id']==k]['doc_full_name'].values)

    # This will render the go.html
    return render_template(
                          'go.html', 
                           query=query,
                           query_art_body=query_art_body, 
                           rec = rec[1:]
                           )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

# this will only be executed when this module is run directly
if __name__ == "__main__":
   main()     
