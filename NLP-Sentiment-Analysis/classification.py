#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import SentimentClassifiers
import RNNSentimentClassifier

# TODO: load training data
list_of_tweets, list_of_emotions = SentimentClassifiers.pre_process_tweets_train_data()
for classifier in ['NaiveBayes', 'SVM', 'logistic', 'RNN']: # You may rename the names of the classifiers to something more descriptive
    trained_model = None
    trained_vector = None
    tokenizer = None
    label_encoder = None
    trained_model_rnn = None
    if classifier == 'NaiveBayes':
        print('Training ' + classifier)
        trained_model, trained_vector = SentimentClassifiers.naive_bayes_classifier_train(list_of_tweets, list_of_emotions)
        # TODO: extract features for training classifier1
        # TODO: train sentiment classifier1
    elif classifier == 'SVM':
        print('Training ' + classifier)
        trained_model, trained_vector = SentimentClassifiers.svm_classifier_train(list_of_tweets,
                                                                                          list_of_emotions)
        # TODO: extract features for training classifier2
        # TODO: train sentiment classifier2
    elif classifier == 'logistic':
        print('Training ' + classifier)
        trained_model, trained_vector = SentimentClassifiers.logistic_classifier_train(list_of_tweets,
                                                                                  list_of_emotions)

        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3
    elif classifier == 'RNN':
        pre_processed_tweets_list = RNNSentimentClassifier.get_training_data()
        print('Training ' + classifier)
        trained_model_rnn, tokenizer, label_encoder = RNNSentimentClassifier.training_rnn_model(pre_processed_tweets_list)



    for testset in testsets.testsets:

        predictions = {}
        # TODO: classify tweets in test set
        if classifier == 'NaiveBayes':
            preprocessedTestDataSet, list_of_test_tweets = SentimentClassifiers.pre_process_tweets_test_data(testset)
            predictions = SentimentClassifiers.naive_bayes_classifier_predict(preprocessedTestDataSet, list_of_test_tweets, trained_model, trained_vector)
        if classifier == 'SVM':
            preprocessedTestDataSet, list_of_test_tweets = SentimentClassifiers.pre_process_tweets_test_data(testset)
            predictions = SentimentClassifiers.svm_classifier_predict(preprocessedTestDataSet,
                                                                              list_of_test_tweets, trained_model,
                                                                              trained_vector)
        if classifier == 'logistic':
            preprocessedTestDataSet, list_of_test_tweets = SentimentClassifiers.pre_process_tweets_test_data(testset)
            predictions = SentimentClassifiers.logistic_classifier_predict(preprocessedTestDataSet,
                                                                              list_of_test_tweets, trained_model,
                                                                              trained_vector)
        if classifier == 'RNN':
            predictions = RNNSentimentClassifier.rnn_predict_model(trained_model_rnn, tokenizer, label_encoder, testset)
        #predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral', '653274888624828198': 'neutral'} # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier

        evaluation.evaluate(predictions, testset, classifier)
        evaluation.confusion(predictions, testset, classifier)
