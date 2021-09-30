import random
import re
from operator import itemgetter

import keras
import numpy as np
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

MAX_VOCAB=5001
MAX_LENGTH = 50


def read_from_file(filepath):
    tweets_initial_list = []
    for line_id, line in enumerate(open(filepath, "r", encoding="utf-8").readlines()):
        columns = line.rstrip().split('\t')
        tweets_initial_list.append(columns)
    return tweets_initial_list

def process_hashtag(sentence):
    words = sentence.split(" ")
    fwords = []
    for word in words:
        if word and len(word)>1 and word[0] == '#':
            word = word[1:]
            mlist = re.findall('[A-Z][^A-Z]*', word)
            fwords.extend(mlist)
        else:
            fwords.append(word)
    fs = ' '.join(fwords)
    return fs

def extend_list(mlist, target_len):
    my_len = len(mlist)
    cnt = my_len
    idx = 0

    while (cnt < target_len):
        mlist.append(mlist[idx])
        idx += 1
        if idx >= my_len:
            idx = 0
        cnt += 1
    return mlist


def pre_process_tweet_data(initial_tweets_list):
    for tweets in initial_tweets_list:
        tweets[2] = process_hashtag(tweets[2])
        tweets[2] = tweets[2].lower()
        tweets[2] = re.sub(
            r"((https?|ftp)://)?[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-z]{2,}[a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*", "",
            tweets[2])  # remove URLs
        tweets[2] = re.sub(r"what's", "what is ", tweets[2])
        tweets[2] = re.sub(r"can't", "can not ", tweets[2])
        tweets[2] = re.sub(r"won't", "will not ", tweets[2])
        tweets[2] = re.sub(r"don't", "do not ", tweets[2])
        tweets[2] = re.sub(r"c'mon", "come on ", tweets[2])
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
        tweets[2] = re.sub(r"[^A-Za-z0-9 ]", " ", tweets[2])
        tweets[2] = re.sub(r"\b[0-9]+\b", "", tweets[2])
        tweets[2] = re.sub(r"\b[a-zA-Z]\b", "", tweets[2])
        tweets[2] = re.sub(r"[ \t]+", " ", tweets[2])  # replace all whitespace characters

    count = 1
    stop_list = set(stopwords.words("english"))
    for tweets in initial_tweets_list:
        tweets[2] = word_tokenize(tweets[2])
        tweets[2] = [word for word in tweets[2] if word not in stop_list]
        tweets[2] = [word for word in tweets[2] if len(word) > 1]
        count += 1
    return initial_tweets_list


def get_weight_matrix(embedding, vocab, top_words):
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((MAX_VOCAB, 100))
    count = 0
    nf = 0
    for word, i in vocab.items():
        count = count + 1
        if count > MAX_VOCAB - 1:
            break
        if word in top_words:
            vector = embedding.get(word)
            if vector is not None:
                weight_matrix[i] = vector

    return weight_matrix


def get_training_data():
    tweets_initial_list = read_from_file('semeval-tweets/twitter-training-data.txt')

    pre_processed_tweets_list = pre_process_tweet_data(tweets_initial_list)
    pos_sentences = []
    neg_sentences = []
    neu_sentences = []
    for items in pre_processed_tweets_list:
        if items[1] == 'positive':
            pos_sentences.append(items)
        if items[1] == 'negative':
            neg_sentences.append(items)
        if items[1] == 'neutral':
            neu_sentences.append(items)
    print('balancing training data...')
    mx_len = max(len(pos_sentences), len(neu_sentences), len(neg_sentences))
    pos_sentences = extend_list(pos_sentences, mx_len)
    neu_sentences = extend_list(neu_sentences, mx_len)
    neg_sentences = extend_list(neg_sentences, mx_len)
    balanced_training_list = []
    balanced_training_list.extend(neu_sentences)
    balanced_training_list.extend(pos_sentences)
    balanced_training_list.extend(neg_sentences)
    random.shuffle(balanced_training_list)
    pre_processed_tweets_list = balanced_training_list
    return pre_processed_tweets_list

def create_predictions_dict(preprocessedTestDataSet, ClassifierResultLabels):
    predictions = {}
    for i in range(0, len(preprocessedTestDataSet)):
        predictions[f'{preprocessedTestDataSet[i][0]}'] = ClassifierResultLabels[i][0]
    return predictions

def training_rnn_model(pre_processed_tweets_list):
    training_words = []
    training_labels = []
    for i in pre_processed_tweets_list:
        training_words.append(i[2])
        training_labels.append(i[1])


    training_sentences = []
    for words in training_words:
        training_sentences.append(" ".join(words))

    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    tokenizer.fit_on_texts(training_sentences)

    encoded_docs = tokenizer.texts_to_sequences(training_sentences)
    X_Train = pad_sequences(encoded_docs, maxlen=MAX_LENGTH)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(training_labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y_Train = onehot_encoder.fit_transform(integer_encoded)

    filename = 'glove.6B.100d.txt'
    with open(filename) as glove_file:
        lines = glove_file.readlines()

    glove_embeddings = {}
    for line in lines:
        parts = line.split()
        glove_embeddings[parts[0]] = np.asarray(parts[1:], dtype='float32')

    sortedWords = sorted(tokenizer.word_counts.items(), key=itemgetter(1), reverse=True)
    top_words = []
    count = 0
    for (word, i) in sortedWords:
        count += 1
        if count > MAX_VOCAB - 1:
            break
        top_words.append(word)

    embedding_vectors = get_weight_matrix(glove_embeddings, tokenizer.word_index, top_words)

    embedding_layer = Embedding(MAX_VOCAB, 100, weights=[embedding_vectors], input_length=MAX_LENGTH, trainable=False)
    new_model = Sequential()
    new_model.add(embedding_layer)
    new_model.add(LSTM(15, dropout=0.3))
    new_model.add(Dense(3, activation='softmax'))
    new_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())

    es_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=4,
                                                restore_best_weights=True)
    new_model.fit(X_Train, Y_Train, epochs=40, verbose=0, callbacks=[es_callback],
                            validation_split=0.2)

    return new_model, tokenizer, label_encoder

def rnn_predict_model(new_model, tokenizer, label_encoder, test_path):
    print("Preprocessing test data...")
    testing_data_path = test_path
    tweets_initial_list_test = read_from_file(f'semeval-tweets/{testing_data_path}')
    pre_processed_test_tweets_list = pre_process_tweet_data(tweets_initial_list_test)
    testing_words = []
    testing_labels = []
    for i in pre_processed_test_tweets_list:
        testing_words.append(i[2])
        testing_labels.append(i[1])


    test_sentences = []
    for words in testing_words:
        test_sentences.append(" ".join(words))

    encoded__test_docs = tokenizer.texts_to_sequences(test_sentences)
    X_Test = pad_sequences(encoded__test_docs, maxlen=MAX_LENGTH)
    predicted_values = new_model.predict(X_Test)

    predicted_labels = []
    for i in range(len(predicted_values)):
        predicted_labels.append(label_encoder.inverse_transform([np.argmax(predicted_values[i])]))

    predictions = create_predictions_dict(tweets_initial_list_test, predicted_labels)
    return predictions