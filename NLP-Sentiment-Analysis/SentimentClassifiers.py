import csv
import os
import pickle
import re
from string import punctuation

import nltk
import nltk.classify
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def read_from_file(filepath):
    tweets_initial_list = []
    tweets_emotions = []
    for line_id, line in enumerate(open(filepath, "r", encoding="utf-8").readlines()):
        columns = line.rstrip().split('\t')
        tweets_initial_list.append([str(columns[0]), str(columns[1]), str(columns[2])])
        tweets_emotions.append((columns[0], columns[1]))
    return tweets_initial_list, tweets_emotions




def read_from_file1(filepath):
    tweets_initial_list = []
    tweets_emotions = []

    with open(filepath, newline='') as raw_tweets:
        tweet_reader = csv.reader(raw_tweets, delimiter='\t')
        for tweets in tweet_reader:
            tweets_initial_list.append(tweets)
        for tweets in tweets_initial_list:
            tweets_emotions.append((tweets[0], tweets[1]))
    return tweets_initial_list, tweets_emotions



def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def pre_process_tweet_data1(initial_tweets_list):
    tweets_list = []
    for tweets in initial_tweets_list:
        #print(tweets[2])
        tweets[2] = tweets[2].lower()
        tweets[2] = re.sub(
            r"((https?|ftp)://)?[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-z]{2,}[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*", "",
            tweets[2])  # remove URLs
        tweets[2] = re.sub(r"what's", "what is ", tweets[2])
        tweets[2] = re.sub(r"can't", "can not ", tweets[2])
        tweets[2] = re.sub(r"won't", "will not ", tweets[2])
        tweets[2] = re.sub(r"don't", "do not ", tweets[2])
        tweets[2] = re.sub(r"&amp", " and ", tweets[2])
        tweets[2] = re.sub(r"amp", " and ", tweets[2])
        tweets[2] = re.sub(r"\'s", " ", tweets[2])
        tweets[2] = re.sub(r"\'ve", " have ", tweets[2])
        tweets[2] = re.sub(r"n't", " not ", tweets[2])
        tweets[2] = re.sub(r"i'm", "i am ", tweets[2])
        tweets[2] = re.sub(r"\'re", " are ", tweets[2])
        tweets[2] = re.sub(r"\'d", " would ", tweets[2])
        tweets[2] = re.sub(r"\'ll", " will ", tweets[2])
        tweets[2] = re.sub('@[^\s]+', '', tweets[2])  # remove usernames
        tweets[2] = re.sub(r'#([^\s]+)', r'\1', tweets[2])  # remove the # in #hashtag
        tweets[2] = re.sub(r"[^A-Za-z0-9 ]", "", tweets[2]) #remove everything that is not a character or a number or whitespace
        tweets[2] = re.sub(r"\b[0-9]+\b", "", tweets[2]) # remove only digit words
        tweets[2] = re.sub(r"\b[a-zA-Z]\b", "", tweets[2]) # remove single characters
        tweets[2] = re.sub(r"\s", " ", tweets[2])  # replace all whitespace characters
        #print(tweets[2])

    lemmatizer = WordNetLemmatizer()
    count = 1
    stop_list = set(stopwords.words("english"))
    for tweets in initial_tweets_list:
        tokenized_word = word_tokenize(tweets[2])
        tokenized_word = [word for word in tokenized_word if word not in stop_list]
        tokenized_word = [word for word in tokenized_word if word not in list(punctuation)]
        tokenized_word = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokenized_word]
        tweets_list.append((tweets[0], tokenized_word))
        # print('Done processing tweet no ', count)
        count += 1
    return tweets_list


def pre_process_tweet_data(initial_tweets_list):
    tweets_list = []
    for tweets in initial_tweets_list:
        tweets[2] = tweets[2].lower()
        tweets[2] = re.sub(
            r"((https?|ftp)://)?[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-z]{2,}[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*", "",
            tweets[2])  # remove URLs
        tweets[2] = re.sub('@[^\s]+', '', tweets[2])  # remove usernames
        tweets[2] = re.sub(r'#([^\s]+)', r'\1', tweets[2])  # remove the # in #hashtag
        tweets[2] = re.sub(r"\'s", " ", tweets[2])
        tweets[2] = re.sub(r"\'ve", " have ", tweets[2])
        tweets[2] = re.sub(r"n't", " not ", tweets[2])
        tweets[2] = re.sub(r"i'm", "i am ", tweets[2])
        tweets[2] = re.sub(r"\'re", " are ", tweets[2])
        tweets[2] = re.sub(r"\'d", " would ", tweets[2])
        tweets[2] = re.sub(r"\'ll", " will ", tweets[2])
        tweets[2] = re.sub(r"[^A-Za-z0-9 ]", "", tweets[2])
        tweets[2] = re.sub(r"\b[0-9]+\b", "", tweets[2])
        tweets[2] = re.sub(r"\s", " ", tweets[2])  # replace all whitespace characters

    lemmatizer = WordNetLemmatizer()
    count = 1
    stop_list = set(stopwords.words("english"))
    for tweets in initial_tweets_list:
        tokenized_word = word_tokenize(tweets[2])
        tokenized_word = [word for word in tokenized_word if word not in stop_list]
        tokenized_word = [word for word in tokenized_word if word not in list(punctuation)]
        tokenized_word = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokenized_word]
        tweets_list.append((tweets[0], tokenized_word))
        #print('Done processing tweet no ', count)
        count += 1
    return tweets_list


def create_final_training_set(tweets_emotions, tweets_list):
    training_tweet_list = []
    training_tweet_id_list = []
    for (i, j) in tweets_emotions:
        for (l, m) in tweets_list:
            if i == l:
                training_tweet_list.append((m, l))
                break

    return training_tweet_list

def create_predictions_dict(preprocessedTestDataSet, ClassifierResultLabels):
    predictions = {}
    for i in range(0, len(preprocessedTestDataSet)):
        predictions[f'{preprocessedTestDataSet[i][1]}'] = ClassifierResultLabels[i]
    return predictions

def identity_tokenizer(text):
    return text



def pre_process_tweets_train_data():
    print("Preprocessing tweets...")
    training_data_path = 'semeval-tweets/twitter-training-data.txt'
    tweets_initial_list, tweets_emotions = read_from_file(training_data_path)
    pre_processed_tweets_list = pre_process_tweet_data(tweets_initial_list)
    training_tweet_list = create_final_training_set(tweets_emotions, pre_processed_tweets_list)
    list_of_tweets = []
    for (tweet, emotion) in training_tweet_list:
        list_of_tweets.append(tweet)

    list_of_emotions = []
    for (tweet, tweet_id1) in training_tweet_list:
        for (tweet_id2, emotion) in tweets_emotions:
            if tweet_id1 == tweet_id2:
                list_of_emotions.append(emotion)
                break
    return list_of_tweets, list_of_emotions


def pre_process_tweets_test_data(test_file_path):
    print("Preprocessing test data...")
    testing_data_path = f'semeval-tweets/{test_file_path}'
    tweets_initial_list_test, tweets_emotions_test = read_from_file(testing_data_path)
    pre_processed_test_tweets_list = pre_process_tweet_data(tweets_initial_list_test)
    preprocessedTestDataSet = create_final_training_set(tweets_emotions_test, pre_processed_test_tweets_list)

    list_of_test_tweets = []
    for (tweet, emotion) in preprocessedTestDataSet:
        list_of_test_tweets.append(tweet)
    return preprocessedTestDataSet, list_of_test_tweets


def naive_bayes_classifier_train(list_of_tweets, list_of_emotions):
    count_vect = CountVectorizer(max_features=5000, tokenizer=identity_tokenizer, min_df=30, lowercase=False)
    X_train_counts = count_vect.fit_transform(list_of_tweets)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print("Training Naive Bayes Model...")
    model = MultinomialNB().fit(X_train_tfidf, list_of_emotions)
    return model, count_vect


def naive_bayes_classifier_predict(preprocessedTestDataSet,list_of_test_tweets, model, count_vect):
    prediction_mnnb = model.predict(count_vect.transform(list_of_test_tweets))
    predictions = create_predictions_dict(preprocessedTestDataSet, prediction_mnnb)
    return predictions

def svm_classifier_train(list_of_tweets, list_of_emotions):
    count_vect = CountVectorizer(tokenizer=identity_tokenizer, min_df=2, lowercase=False)
    X_train_counts = count_vect.fit_transform(list_of_tweets)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    model = LinearSVC().fit(X_train_tfidf, list_of_emotions)
    return model, count_vect


def logistic_classifier_train(list_of_tweets, list_of_emotions):
    count_vect = CountVectorizer(tokenizer=identity_tokenizer, min_df=2, lowercase=False)
    X_train_counts = count_vect.fit_transform(list_of_tweets)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    model = LogisticRegression(max_iter=5000).fit(X_train_tfidf, list_of_emotions)
    return model, count_vect


def svm_classifier_predict(preprocessedTestDataSet, list_of_test_tweets, model, count_vect):
    prediction_svm = model.predict(count_vect.transform(list_of_test_tweets))
    predictions = create_predictions_dict(preprocessedTestDataSet, prediction_svm)
    return predictions


def logistic_classifier_predict(preprocessedTestDataSet, list_of_test_tweets, model, count_vect):
    prediction_logistic = model.predict(count_vect.transform(list_of_test_tweets))
    predictions = create_predictions_dict(preprocessedTestDataSet, prediction_logistic)
    return predictions
