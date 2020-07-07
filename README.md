# Recommendation system using python

In this project, we work with datasets containing information about the interactions that users have with articles on the IBM Watson Studio platform. We use these information to make article recommendations to users. The project also includes a web app where a user can read an article by entering it's name and also the app will suggest more articles in the related context that the user might want to read.

## Requirements

The following packages are required to run the code :
- **Python 3.7+**

- **NumPy** (`pip install numpy`)

- **pandas** (`pip install pandas`)

- **nltk** (`pip install nltk`)

- **scikit-learn** (`pip install scikit-learn`)

-  **flask** (`pip install flask`)

In order to be able to use nltk packages used in this code, open a python shell and run the following to download nltk data :

1. `import nltk`

2. `nltk.download()`

A pop-up window will appear and then click on download. Alternatively, you might also want to download only the libraries that will be needed to run this code. For this, in the python shell, run the following command :

`nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])`

## List of files

- _**articles_community.csv**_ : This csv file contains information about articles available on the IBM Watson Studio platform.

- _**user-item-interactions.csv**_ : This file has the data related to how users interact with articles on this platform.

- _**Recommendations_with_IBM.ipynb**_ : This is a jupyter notebook with a detail explanation of the analysis.

- _**recommender_functions.py**_ : A python module containing all the necessary functions used in the analysis.

- _**recommender.py**_ : A python module summing up all the necessary steps to build recommendation engine into a single and ready-to-use class.

- _**run.py**_  :  The python file responsible for the back-end of the web app.

- _**homepg.html**_ : The html file that renders the homepage of the web app.

- _**go.html**_ : The html file that renders the query page of the web app where a user can read about an article and also find articles of similar content.


## Analysis method

The whole analysis can be broken down into four main steps :

- **Exploratory Data Analysis**

   In this step, we take a look at both our datasets closely and perform data cleaning whereever required.

- **Rank based recommendations**

  In this type of recommendation engine, we find the most popular articles simply based on the number of interactions they have with users on the IBM Watson Studio platform. These are then the articles we might recommend to new users or anyone depending on how much we know about them.

- **User-user based collaborative filtering**

  In this type of recommendation technique, we look for users who are closely related to each other in terms of their interactions with items, *e.g.*, users who have watched same movies and have given ratings in the same ballpark are *neighbors* in the movie ratings plane. Using the information about neighbors of a user, we can then recommend items to that user.

- **Content based recommendations**

  In this technique, we use features or information about items, users to provide recommendations, *e.g.*, recommending movies on the basis of its genres is an example of content based recommendation.

  The last two methods provide more personalised recommendations to users as compared to rank based recommendations.


## Run the app

To run the app locally on your machine, follow the below steps:

1. To run the python script file

     `python recommender.py data/user-item-interactions.csv data/articles_community.csv`

2. To run the web app locally

    `python run.py`

 4. Go to http://0.0.0.0:3001/

## External link

The web app is publicly available on the [heroku cloud server](https://article-recommendation-app.herokuapp.com).
