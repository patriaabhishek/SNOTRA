# Importing libraries
from flask import Flask, flash, redirect, render_template, request, session, abort,send_from_directory,send_file,jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime
from surprise import Reader, Dataset
from surprise.prediction_algorithms import knns
import nltk
import string
import scipy as sp
import warnings

warnings.filterwarnings('ignore')

##############################################################
##
## Data Preparation
##
##############################################################
         
DATA_FOLDER       = 'data/'   
USERS_FILE        = 'users.csv'
#USERS_FILE = "user_reviews_1.csv"
REVIEWS_FILE      = 'reviews.csv'
BOOKS_FILE        = 'books.csv'

newdatasetFlag = (USERS_FILE == "users.csv")

#READ CSV
def read_csv(file, encoding = ''):
    file_path = DATA_FOLDER + file
    if len(encoding):
        df = pd.read_csv(file_path, encoding = encoding) 
    else:
        df = pd.read_csv(file_path)
    return df

# Reading data from the CSV
def get_data():
  ratings_data_raw = read_csv(USERS_FILE)
  return ratings_data_raw

def get_clean_data():
  ratings_data_raw = get_data()
  ratings_data = data_cleanup(ratings_data_raw)
  return ratings_data

#Data cleanup
def data_cleanup(ratings_data_raw):
  #remove nulls and 0 entry rows
  if newdatasetFlag == True:
    ratings_data = ratings_data_raw[(ratings_data_raw['rating'] != 0) & (ratings_data_raw['rating'] != '')]
    
    #remove unnecessary columns
    del ratings_data['Unnamed: 0']
    del ratings_data['book_name']
    del ratings_data['average_rating']
  
  else:
    
    ratings_data = ratings_data_raw[(ratings_data_raw['Rating'] != 0) & (ratings_data_raw['Rating'] != '')]
    
    #remove unnecessary columns
    del ratings_data['Book_name']
    del ratings_data['Image_url']

    #rename columns
    ratings_data.rename(columns = {'Rating':'rating', 'User_id':'user_id', 'Book_id':'book_id'}, inplace = True) 

  return ratings_data


##############################################################
##
## Collaborative Filtering
##
##############################################################

#Build the tuples of ratings, book_id and user_id
def buildMatrix(df):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
  
    print(current_time,"===========Building Matrix===============\n")
  
    df = df.pivot(index='user_id', columns='book_id', values='rating')
    df = df.fillna(0)

    user_id_index_temp = list(df.index)
    user_id_index = {}
    
    for count, each in enumerate(user_id_index_temp, 0):
        user_id_index[int(each)] = count

    index_book_id_temp = list(df.columns)
    index_book_id = {}

    for count, each in enumerate(index_book_id_temp, 0):
        index_book_id[count] = int(each)

    return (df, user_id_index, index_book_id)

#this is where magic happens, total crazzyyyy
def cache_matrix(reInitialize = False, fetchDataFromCsv = True):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time, "===========Caching Matrix===============\n")
    
    if reInitialize or 'algo' not in globals() or 'ratings_matrix' not in globals():
      global ratings_matrix 
      global user_id_index
      global index_book_id
      global ratings_data
      global algo

      reader = Reader(rating_scale=(0.5, 5.0))
      data = Dataset.load_from_df(ratings_data[['user_id', 'book_id', 'rating']], reader)
      dataset = data.build_full_trainset()
      algo = knns.KNNBaseline(k=27,sim_options = {'name':'cosine','user_based':True})
      algo.fit(dataset)

      if fetchDataFromCsv == True:
         ratings_data_raw = get_data()
         ratings_data = data_cleanup(ratings_data_raw)

      user_rating_matrix = buildMatrix(ratings_data)
      ratings_matrix = user_rating_matrix[0]
      user_id_index = user_rating_matrix[1]
      index_book_id = user_rating_matrix[2]

#Calculate the cosine similarity metric
def calc_cosine_similarity(vA,vB):
    sumA = np.sqrt(np.sum(np.multiply(vA,vA)))
    sumB = np.sqrt(np.sum(np.multiply(vB,vB)))
    sumAB = np.sum(np.multiply(vA,vB))
    cosine_similarity = sumAB / (sumA * sumB)

    return cosine_similarity

#Get similar users for a given user
def get_similar_users(ratings, user_id_index, current_user_id, topN):
    selected_row = np.array(ratings[ratings.index == current_user_id])
    user_dict = {}

    for index, row in ratings.iterrows():
        if(index != current_user_id):
            user_dict[index] = calc_cosine_similarity(selected_row,np.array(row))            
    
    user_dict = sorted(user_dict.items(), key=lambda x:x[1], reverse=True)
    user_dict = user_dict[:topN]
    
    top_user_ids = [i for i,j in user_dict]

    return top_user_ids

#Gets the recommended books from a similar user
def recommend_book_from_sim_user(ratings, similar_user_ids, user_id_index, index_book_id, topN):
  books = []  
  for user_id in similar_user_ids:
      selected_book_ratings = ratings_matrix.loc[user_id]
      selected_book_ratings.sort_values(ascending=False, inplace = True)
      books.extend(list(selected_book_ratings[:topN].index))

  books = np.unique(books)[:5]    
  return books

#this is the exposed end point, returns the recommended books for a given user
def get_recommended_books(current_user_id, reInitialize = False, topN = 5, topBooks = 2):  
  
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  print(current_time, "===========Getting recommended books===============\n")
  cache_matrix(reInitialize)
  similar_users = get_similar_users(ratings_matrix, user_id_index, current_user_id, topN)
  return recommend_book_from_sim_user(ratings_matrix, similar_users, user_id_index, index_book_id, topBooks)

##############################################################
##
## Hybrid Filtering
##
##############################################################
def collaborative_prediction(userid,ratings_data,n):
    
    user_read_books = ratings_data[ratings_data["user_id"]==userid]["book_id"].values
    not_user_read_books = ratings_data[~ratings_data["book_id"].isin(user_read_books)] #all bookids that user has not read
    book_ids = not_user_read_books["book_id"].values
    book_rating = {}
    for book in book_ids:
        estimated_rating = algo.predict(userid,book).est
        book_rating[book] = estimated_rating
    book_rating = sorted(book_rating.items(), key=lambda x:x[1], reverse=True)
    book_rating = book_rating[:n]
    ids = [i for i,r in book_rating]
    return ids


#Data cleanup
def data_cleanup_content_based(description_data_raw):
  description_data = description_data_raw.copy()
  #remove unnecessary columns
  del description_data_raw['Unnamed: 0']
  del description_data_raw['book_name']
  #filtering ratings  with a threshold
  description_data_raw = description_data_raw[description_data_raw['average_rating']>4.5]
  #deleting ratings column
  del description_data_raw['average_rating']
  #renaming columns
  description_data_raw.columns = ['book_description', 'book_id']
  #Removing na and duplicate values
  description_data_raw = description_data_raw.dropna(axis = 0)
  description_data_raw = description_data_raw.drop_duplicates(subset='book_id', keep='last')
  description_data_raw.set_index('book_id', inplace = True)
  description_data = description_data_raw
  return description_data


#Removing HTML tags and special characters

def remove_html_tags(text):
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def clean_text(text):
    text = text.lower()
    new_text =  "".join([i if i.isalpha() else ' ' for i in text])
    #return "".join([i for i in text if i.isspace()])
    return new_text

def process_descriptions(description_data):
  #remove_html_tags(data.loc[0][1])
  description_data['book_description'] = description_data['book_description'].apply(lambda x: remove_html_tags(x))
  #Lowercasing
  description_data['book_description'] = description_data['book_description'].apply(lambda x: x.lower())
  #remove_numbers_special_characters
  description_data['book_description'] = description_data['book_description'].apply(lambda x: clean_text(x))
  return description_data


def create_stop_words():
  #Creating an exhaustive list of stopwords
  nltk.download('stopwords')
  nltk.download('wordnet')
  sw1 = set(nltk.corpus.stopwords.words('english'))
  sw2 = set(string.ascii_lowercase)
  sw3 = set(["a's" , "able" , "about" , "above" , "according" , "accordingly" , "across" , "actually" , "after" , "afterwards" , "again" , "against" , "ain't" , "all" , "allow" , "allows" , "almost" , "alone" , "along" , "already" , "also" , "although" , "always" , "am" , "among" , "amongst" , "an" , "and" , "another" , "any" , "anybody" , "anyhow" , "anyone" , "anything" , "anyway" , "anyways" , "anywhere" , "apart" , "appear" , "appreciate" , "appropriate" , "are" , "aren't" , "around" , "as" , "aside" , "ask" , "asking" , "associated" , "at" , "available" , "away" , "awfully" , "be" , "became" , "because" , "become" , "becomes" , "becoming" , "been" , "before" , "beforehand" , "behind" , "being" , "believe" , "below" , "beside" , "besides" , "best" , "better" , "between" , "beyond" , "both" , "brief" , "but" , "by" , "c'mon" , "c's" , "came" , "can" , "can't" , "cannot" , "cant" , "cause" , "causes" , "certain" , "certainly" , "changes" , "clearly" , "co" , "com" , "come" , "comes" , "concerning" , "consequently" , "consider" , "considering" , "contain" , "containing" , "contains" , "corresponding" , "could" , "couldn't" , "course" , "currently" , "definitely" , "described" , "despite" , "did" , "didn't" , "different" , "do" , "does" , "doesn't" , "doing" , "don't" , "done" , "down" , "downwards" , "during" , "each" , "edu" , "eg" , "eight" , "either" , "else" , "elsewhere" , "enough" , "entirely" , "especially" , "et" , "etc" , "even" , "ever" , "every" , "everybody" , "everyone" , "everything" , "everywhere" , "ex" , "exactly" , "example" , "except" , "far" , "few" , "fifth" , "first" , "five" , "followed" , "following" , "follows" , "for" , "former" , "formerly" , "forth" , "four" , "from" , "further" , "furthermore" , "get" , "gets" , "getting" , "given" , "gives" , "go" , "goes" , "going" , "gone" , "got" , "gotten" , "greetings" , "had" , "hadn't" , "happens" , "hardly" , "has" , "hasn't" , "have" , "haven't" , "having" , "he" , "he's" , "hello" , "help" , "hence" , "her" , "here" , "here's" , "hereafter" , "hereby" , "herein" , "hereupon" , "hers" , "herself" , "hi" , "him" , "himself" , "his" , "hither" , "hopefully" , "how" , "howbeit" , "however" , "i'd" , "i'll" , "i'm" , "i've" , "ie" , "if" , "ignored" , "immediate" , "in" , "inasmuch" , "inc" , "indeed" , "indicate" , "indicated" , "indicates" , "inner" , "insofar" , "instead" , "into" , "inward" , "is" , "isn't" , "it" , "it'd" , "it'll" , "it's" , "its" , "itself" , "just" , "keep" , "keeps" , "kept" , "know" , "known" , "knows" , "last" , "lately" , "later" , "latter" , "latterly" , "least" , "less" , "lest" , "let" , "let's" , "like" , "liked" , "likely" , "little" , "look" , "looking" , "looks" , "ltd" , "mainly" , "many" , "may" , "maybe" , "me" , "mean" , "meanwhile" , "merely" , "might" , "more" , "moreover" , "most" , "mostly" , "much" , "must" , "my" , "myself" , "name" , "namely" , "nd" , "near" , "nearly" , "necessary" , "need" , "needs" , "neither" , "never" , "nevertheless" , "new" , "next" , "nine" , "no" , "nobody" , "non" , "none" , "noone" , "nor" , "normally" , "not" , "nothing" , "novel" , "now" , "nowhere" , "obviously" , "of" , "off" , "often" , "oh" , "ok" , "okay" , "old" , "on" , "once" , "one" , "ones" , "only" , "onto" , "or" , "other" , "others" , "otherwise" , "ought" , "our" , "ours" , "ourselves" , "out" , "outside" , "over" , "overall" , "own" , "particular" , "particularly" , "per" , "perhaps" , "placed" , "please" , "plus" , "possible" , "presumably" , "probably" , "provides" , "que" , "quite" , "qv" , "rather" , "rd" , "re" , "really" , "reasonably" , "regarding" , "regardless" , "regards" , "relatively" , "respectively" , "right" , "said" , "same" , "saw" , "say" , "saying" , "says" , "second" , "secondly" , "see" , "seeing" , "seem" , "seemed" , "seeming" , "seems" , "seen" , "self" , "selves" , "sensible" , "sent" , "serious" , "seriously" , "seven" , "several" , "shall" , "she" , "should" , "shouldn't" , "since" , "six" , "so" , "some" , "somebody" , "somehow" , "someone" , "something" , "sometime" , "sometimes" , "somewhat" , "somewhere" , "soon" , "sorry" , "specified" , "specify" , "specifying" , "still" , "sub" , "such" , "sup" , "sure" , "t's" , "take" , "taken" , "tell" , "tends" , "th" , "than" , "thank" , "thanks" , "thanx" , "that" , "that's" , "thats" , "the" , "their" , "theirs" , "them" , "themselves" , "then" , "thence" , "there" , "there's" , "thereafter" , "thereby" , "therefore" , "therein" , "theres" , "thereupon" , "these" , "they" , "they'd" , "they'll" , "they're" , "they've" , "think" , "third" , "this" , "thorough" , "thoroughly" , "those" , "though" , "three" , "through" , "throughout" , "thru" , "thus" , "to" , "together" , "too" , "took" , "toward" , "towards" , "tried" , "tries" , "truly" , "try" , "trying" , "twice" , "two" , "un" , "under" , "unfortunately" , "unless" , "unlikely" , "until" , "unto" , "up" , "upon" , "us" , "use" , "used" , "useful" , "uses" , "using" , "usually" , "value" , "various" , "very" , "via" , "viz" , "vs" , "want" , "wants" , "was" , "wasn't" , "way" , "we" , "we'd" , "we'll" , "we're" , "we've" , "welcome" , "well" , "went" , "were" , "weren't" , "what" , "what's" , "whatever" , "when" , "whence" , "whenever" , "where" , "where's" , "whereafter" , "whereas" , "whereby" , "wherein" , "whereupon" , "wherever" , "whether" , "which" , "while" , "whither" , "who" , "who's" , "whoever" , "whole" , "whom" , "whose" , "why" , "will" , "willing" , "wish" , "with" , "within" , "without" , "won't" , "wonder" , "would" , "wouldn't" , "yes" , "yet" , "you" , "you'd" , "you'll" , "you're" , "you've" , "your" , "yours" , "yourself" , "yourselves" , "zero"])
  global stopwords 
  stopwords = sw3.union(sw2.union(sw1))
 # return stopwords

def extract_words(doc, stopwords):
    from nltk.stem import WordNetLemmatizer 
    lemmatizer = WordNetLemmatizer() 
    new_doc = str(doc).split()
    return [lemmatizer.lemmatize(i) for i in new_doc if i not in stopwords]


def content_book_ids_and_description(description_data):
  #Creating a dictionary of descriptions with each word separated and stopwords removed
  global book_ids 
  global book_description
  book_ids = sorted(list(description_data.index))
  book_description = {}
  for each in book_ids:
      book_description[each] = extract_words(description_data.loc[each]['book_description'], stopwords)
    
def vocabulary_list(book_ids, book_description):
  global book_description_list 
  book_description_list = []
  for i in book_ids:
      book_description_list.append(' '.join(book_description[i]))
        
def content_vectorizer(book_description_list):
  from sklearn.feature_extraction.text import CountVectorizer
  vectorizer = CountVectorizer()
  global bow
  bow = vectorizer.fit_transform(book_description_list)
    

def similarity_matrix_generator(bow):
  #Normalizing bag of words
  ssq =  bow.power(2).sum(axis = 1)
  temp = (1/ssq)
  bow_norm = bow.multiply(temp)
  #Converting COO to CSR
  bow_norm = bow_norm.tocsr()
  bow_norm_transpose = bow_norm.transpose()
  #Generating similarity matrix
  global similarity_matrix
  similarity_matrix = bow_norm.dot(bow_norm_transpose)
    


def content_based_similar_books(book_ids, similarity_matrix, input_book_id=1, top_k_suggestions=5):
    
  if input_book_id in book_ids:
    #Getting similar books
    similar_books_sparse = similarity_matrix[book_ids.index(input_book_id)].toarray()
    similar_books_list = list(enumerate(similar_books_sparse[0], 0))
    similar_books_sorted_by_score = sorted(similar_books_list, key = lambda x: x[1], reverse = True)
    #Generating list of top k similar books
    top_k_similar_books = [i[0] for i in similar_books_sorted_by_score[0:top_k_suggestions+1]]
    top_k_similar_book_ids = [book_ids[i] for i in top_k_similar_books]
    #Removing the book id asked
    if input_book_id in top_k_similar_book_ids:
      top_k_similar_book_ids.remove(input_book_id)
    #Returning list of similar books
    return top_k_similar_book_ids[:top_k_suggestions]
  
  else:
    return []

def content_prediction(user_id, ratings_data, book_ids, similarity_matrix, n = 5):
  books = list(ratings_data[ratings_data['user_id'] == user_id]['book_id'])
  recommended_books_set = set()
  for book in books:
    temp = set(content_based_similar_books(book_ids, similarity_matrix, book, n))
    recommended_books_set = recommended_books_set.union(temp)
  recommended_books = list(recommended_books_set)
  return recommended_books

def hybrid(user_id,ratings_data,book_ids,similarity_matrix):

    
    content_pred = set(content_prediction(user_id, ratings_data, book_ids, similarity_matrix))
    collaborative_pred = collaborative_prediction(user_id,ratings_data, 10)
    col_pred = set(collaborative_pred)
    intersect_pred = content_pred.intersection(col_pred)
    pred = list(intersect_pred)[:5]
    n = (5-len(intersect_pred))//2
    if(n>0):
        if(len(content_pred)>=n):
            content_pred = list(content_pred)[:n]
            
        n = 5-len(intersect_pred)-len(content_pred)
        collaborative_pred = list(collaborative_pred)[:n]
        pred = pred+list(content_pred)+collaborative_pred
    return pred

ratings_data = get_clean_data()
description_data = data_cleanup_content_based(read_csv(REVIEWS_FILE))
description_data = process_descriptions(description_data)


create_stop_words()
content_book_ids_and_description(description_data)
vocabulary_list(book_ids, book_description)
content_vectorizer(book_description_list)

if 'similarity_matrix' not in globals():
  global similarity_matrix
  similarity_matrix = sp.sparse.load_npz('./similarity_matrix.npz')

def get_hybrid_recommended_books(current_user_id, reInitialize = False, topN = 5, topBooks = 1):  
  
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  print(current_time, "===========Getting recommended books===============\n")
  cache_matrix(reInitialize)
  #similar_users = get_similar_users(ratings_matrix, user_id_index, current_user_id, topN)
  newdatasetFlag = True
  #print("Cleaned Dataset size: ", len(ratings_data))
  global ratings_data
  ratings_data.reset_index(inplace=True)
  ratings_data = ratings_data.drop(columns=['index'])
  return hybrid(current_user_id,ratings_data,book_ids,similarity_matrix)


##############################################################
##
## Helper Functions
##
##############################################################

#creates a map of book_id to image_url
def get_book_url_map():
  reviews_raw = read_csv(USERS_FILE)
  book_url_df = reviews_raw[['Book_id', 'Image_url']]
  book_url_df.rename(columns = {'Image_url':'image_url', 'Book_id':'book_id'}, inplace = True) 
  return book_url_df.set_index('book_id').T.to_dict()

#creates a map of book_id to book_name
def get_book_name_map():
  reviews_raw = read_csv(USERS_FILE)
  book_url_df = reviews_raw[['Book_id', 'Book_name']]
  book_url_df.rename(columns = {'Book_name':'book_name', 'Book_id':'book_id'}, inplace = True) 
  return book_url_df.set_index('book_id').T.to_dict()

#returns the avarage rating of a book
def get_average_book_rating():
  reviews_raw = read_csv(USERS_FILE)
  book_url_df = reviews_raw[['Book_id', 'Rating']]
  book_url_df.rename(columns = {'Rating':'rating', 'Book_id':'book_id'}, inplace = True) 
  return book_url_df.groupby('book_id').mean()

#Get n random books to show a new user 
def get_random_books(n=10): 
  books_raw = read_csv(BOOKS_FILE)
  book_df = books_raw[['book_id','author', 'book_url', 'book_name', 'genre', 'image_url', 'average_rating']]
  book_df.rename(columns = {'book_name':'book_name_long'}, inplace = True) 
  book_df['book_id_copy'] = book_df['book_id']
  book_df['book_name'] = book_df['book_name_long'].astype(str).str[:30]
  book_df['author'] = book_df['author'].astype(str).str[:30]
        
  book_df = book_df.sample(n=n)
  return book_df.set_index('book_id_copy').T.to_dict()


#Crazzzzy max work! I like the design here. I like, I like, I like!
def get_custom_userID(random_books, rating1, rating2, rating3, rating4, rating5, rating6, rating7, rating8, rating9, rating10):
  
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  print(current_time, "===========Generating New User===============\n")
  
  global ratings_data
  new_user_id = int(ratings_data[['user_id']].max().user_id + 1)

  random_books = pd.DataFrame(random_books).T.reset_index()
  
  new_data = list()
  new_data.append({"book_id":random_books.iloc[0].book_id, "rating":int(rating1), "user_id":new_user_id})
  new_data.append({"book_id":random_books.iloc[1].book_id, "rating":int(rating2), "user_id":new_user_id})
  new_data.append({"book_id":random_books.iloc[2].book_id, "rating":int(rating3), "user_id":new_user_id})
  new_data.append({"book_id":random_books.iloc[3].book_id, "rating":int(rating4), "user_id":new_user_id})
  new_data.append({"book_id":random_books.iloc[4].book_id, "rating":int(rating5), "user_id":new_user_id})
  new_data.append({"book_id":random_books.iloc[5].book_id, "rating":int(rating6), "user_id":new_user_id})
  new_data.append({"book_id":random_books.iloc[6].book_id, "rating":int(rating7), "user_id":new_user_id})
  new_data.append({"book_id":random_books.iloc[7].book_id, "rating":int(rating8), "user_id":new_user_id})
  new_data.append({"book_id":random_books.iloc[8].book_id, "rating":int(rating9), "user_id":new_user_id})
  new_data.append({"book_id":random_books.iloc[9].book_id, "rating":int(rating10), "user_id":new_user_id})
  new_df = pd.DataFrame(new_data)

  ratings_data = pd.concat([ratings_data, new_df])
  cache_matrix(reInitialize = True, fetchDataFromCsv = False)

  return new_user_id

#gets all the information we've for each book  
def get_book_info_map():
  books_raw = read_csv(BOOKS_FILE)
  book_info_df = books_raw[['book_id', 'author', 'book_url', 'book_name', 'genre', 'image_url', 'average_rating']]
  #book_info_df["image_url"] = book_info_df["book_url"].str.replace("/show/", "/photo/")
  return book_info_df.set_index('book_id').T.to_dict()

  #returns the user data
def get_user_info():
  user_data_raw = read_csv(USERS_FILE, encoding='iso-8859-1')
  user_data = user_data_raw

  del user_data["user_reviews_count"]
  return user_data.set_index('user_id').T.to_dict()


##############################################################
##
## Force Graph Helper Functions
##
##############################################################
def get_force_graph(UserId, books):
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  print(current_time, "===========Getting Force Graph Data===============\n")
  user_info = get_user_info()
  books_info = get_book_info_map()

  cache_matrix(False)
  similar_users = get_similar_users(ratings_matrix, user_id_index, int(UserId), 5)

  nodes = [];
  labels = [];

  try:
    Username = user_info[UserId]["name"]
  except:
    Username = "George P. Burdell"
  
  #person_url = "https://imagesvc.meredithcorp.io/v3/mm/image?url=https://static.onecms.io/wp-content/uploads/sites/24/2018/12/gettyimages-919870926-2000.jpg"
  person_url = "https://image.freepik.com/free-vector/man-avatar-profile-round-icon_24640-14044.jpg"
  person_url2 = "https://image.freepik.com/free-vector/woman-avatar-profile-round-icon_24640-14042.jpg"
  person_url3 = "https://www.goodreads.com/"
  
  nodes.append({"id" : UserId, "name" : Username, "group":1, "is_user":1, "image_url": person_url, "book_url": person_url3})
  for xx, book_id in enumerate(books):
    book_id_2 = "1_" + str(xx) + "_" + str(book_id)
    nodes.append({"id" : book_id_2, "name" : books_info[book_id]["book_name"][:30], "group":1, "is_user":0, "image_url":books_info[book_id]["image_url"], "book_url":books_info[book_id]["book_url"]})
    labels.append({"source": UserId, "target": book_id_2, "value": 20})
    
  for idx, similar_user in enumerate(similar_users[:2]):
      
    try:
      Username = user_info[similar_user]["name"]
    except:
      Username = "George P. Burdell"


    similar_user_books = get_hybrid_recommended_books(similar_user)
    labels.append({"source": UserId, "target": similar_user, "value": 30*(idx+2)})
    nodes.append({"id" : similar_user, "name" : Username, "group":idx+2, "is_user":1, "image_url": person_url2, "book_url":person_url3})
    for xx, book_id in enumerate(similar_user_books):
      book_id_2 = str(idx+2) + "_" + str(xx) + "_" + str(book_id)
      labels.append({"source": similar_user, "target": book_id_2, "value": 20})
      nodes.append({"id" : book_id_2, "name" : books_info[book_id]["book_name"][:30], "group":idx+2, "is_user":0, "image_url":books_info[book_id]["image_url"], "book_url":books_info[book_id]["book_url"]})

  # data2 = pd.DataFrame(nodes).set_index('id').T.to_dict()
  # force_graph = {"xx":{"nodes":nodes, "links":labels, "data2": data2 }}
  # data2 = pd.DataFrame(nodes).set_index('id').T.to_dict()
  help_df = pd.DataFrame(nodes)
  help_data = help_df.set_index("id").T.to_dict()
  #force_graph = {"xx": {"nodes":nodes, "links":labels, "help_data":help_data }}
  force_graph = {"xx": {"nodes":nodes, "links":labels }}
  
  return force_graph


#############################################################
##
## FLASK
##
#############################################################

application= Flask(__name__)

class DataStore():
    UserID=None
    
data=DataStore()

@application.route("/main",methods=["GET","POST"])

#3. Define main code
@application.route("/",methods=["GET","POST"])

def homepage():
    if 'ratings_data' not in globals():
      global ratings_data
      ratings_data = get_clean_data()
    
    random_books = get_random_books()
    if newdatasetFlag == True:
      default_user_id = 16281068
    else:
      default_user_id = 5808559

    UserId   = request.form.get('UserId', default_user_id)
  
    rating1  = rating2 = rating3 = rating4 = rating5 = rating6 = rating7 = rating8 = rating9 = rating10 = 0
    rating1  = request.form.get('Rating1',  0)
    rating2  = request.form.get('Rating2',  0)
    rating3  = request.form.get('Rating3',  0)
    rating4  = request.form.get('Rating4',  0)
    rating5  = request.form.get('Rating5',  0)
    rating6  = request.form.get('Rating6',  0)
    rating7  = request.form.get('Rating7',  0)
    rating8  = request.form.get('Rating8',  0)
    rating9  = request.form.get('Rating9',  0)
    rating10 = request.form.get('Rating10', 0)

    if(rating1 != 0 or rating2 != 0 or rating3 != 0 or rating4 != 0 or rating5 != 0 or rating6 != 0 or rating7 != 0 or rating8 != 0 or rating9 != 0 or rating10 != 0):    
        #print("here 2")
        UserId = get_custom_userID(random_books, rating1, rating2, rating3, rating4, rating5, rating6, rating7, rating8, rating9, rating10)
        #print("new userID", UserId)
    #print(UserId)
        
    data.UserId = UserId
    print("userID", UserId)
    books = get_hybrid_recommended_books(UserId)
    print("Userid: ",UserId, " -- ", books )

    books_info_map = get_book_info_map()
    user_info_map = get_user_info()

    selected_books = books
    
    books = [ [book] for book in books]
    df = pd.DataFrame(books,  columns = ['BookID'])
    flare = dict()

    force_graph = get_force_graph(int(UserId), selected_books)

    d = {"name": "flare", "children": [], "random_books": random_books, "force_graph": force_graph}
    
    for row in df.values:
        try:
          userName = user_info_map[UserId]["name"]
        except : userName = "George P Burdell (New User)"  

        userID = str(UserId)
        bookID = str(row[0])

        ##Get Book information for displaying
        book_name  = books_info_map[row[0]]['book_name'][:30]
        image_url  = books_info_map[row[0]]['image_url']
        author     = books_info_map[row[0]]['author'][:30]
        genre      = books_info_map[row[0]]['genre']
        avg_rating = books_info_map[row[0]]['average_rating']
        book_url   = books_info_map[row[0]]['book_url']

        # make a list of keys
        keys_list = []
        for item in d['children']:
            keys_list.append(item['userID'])

        books_info = {
            "bookID"    : bookID,
            "book_name" : book_name,
            "book_url"  : book_url,
            "image_url" : image_url,
            "author"    : author,
            "genre"     : genre,
            "avg_rating": avg_rating
        }    

        # if 'the_parent' is NOT a key in the flare.json yet, append it
        if not userID in keys_list:
            d['children'].append({"userID": userID, "user_name": userName, "children": [books_info]})

        # if 'the_parent' IS a key in the flare.json, add a new child to it
        else:
            d['children'][keys_list.index(userID)]['children'].append(books_info)

    flare = d
    e = json.dumps(flare)

    data.Prod = json.loads(e)
    Prod = data.Prod

    return render_template("index.html", Prod=Prod)

@application.route("/get-data", methods=[ "GET", "POST" ])
def returnProdData():
    f = data.Prod
    return jsonify(f)
# export the final result to a json file

if __name__ == "__main__":
    application.run( debug = True )
