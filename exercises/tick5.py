from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities
from exercises.tick4 import sign_test
import random

def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    copy_data = training_data.copy()
    list_of_folds = []
    fold_size = len(training_data) // n
    for _ in range(n-1):
        curr_fold = []
        for _ in range(fold_size):
            # generate a random index to take an instance from the training data
            rand_index = random.randint(0,len(copy_data)-1)
            curr_fold.append(copy_data[rand_index])
            # remove the instance from the training data
            copy_data.pop(rand_index)
        list_of_folds.append(curr_fold)
    list_of_folds.append(copy_data)
    return list_of_folds


def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    copy_data = training_data.copy()
    prev_sent = -1
    list_of_folds = []
    fold_size = len(training_data) // n
    for _ in range(n-1):
        curr_fold = []
        for _ in range(fold_size):
            # generate a random index to take an instance from the training data
            rand_index = random.randint(0,len(copy_data)-1)
            # Loop until we get a random review that has different sentiment to the last one we added
            while(copy_data[rand_index]['sentiment'] == prev_sent):
                # generate a random index to take an instance from the training data
                rand_index = random.randint(0,len(copy_data)-1)
            prev_sent = copy_data[rand_index]['sentiment']
            curr_fold.append(copy_data[rand_index])
            # remove the instance from the training data
            copy_data.pop(rand_index)
        list_of_folds.append(curr_fold)
    list_of_folds.append(copy_data)
    return list_of_folds

def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    accuracies = []
    for test_fold in split_training_data:
        copy_data = split_training_data.copy()
        copy_data.remove(test_fold)
        training_data = [review for fold in copy_data for review in fold]
        predictions = []
        true_sentiments = []
        # Calculate the probabilites required for NB 
        log_probabilities = calculate_smoothed_log_probabilities(training_data)
        class_probabilities = calculate_class_log_probabilities(training_data)
        # For each review add the NB classifiers' prediction to a list
        for review in test_fold:
            true_sentiments.append(review['sentiment'])
            predictions.append(predict_sentiment_nbc(review['text'], log_probabilities, class_probabilities))
        # Calculate the accuracy for that fold and append to the list
        accuracies.append(accuracy(predictions, true_sentiments))
    return accuracies


def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    return sum(accuracies) / len(accuracies)


def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    mean = cross_validation_accuracy(accuracies)
    var_sum = 0
    for val in accuracies:
        var_sum += (val-mean)**2
    return 1/len(accuracies) * var_sum


def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    matrix = [[0,0],[0,0]]
    for i in range(len(predicted_sentiments)):
        # If was actually positive
        if actual_sentiments[i] == 1:
            # If prediction postiive
            if predicted_sentiments[i] == 1:
                matrix[0][0] += 1
            # If prediction negative
            else:
                matrix[1][0] += 1
        # If was actually negative
        else:
            # If prediction positive
            if predicted_sentiments[i] == 1:
                matrix[0][1] += 1
            # If prediction negative
            else:
                matrix[1][1] += 1
    return matrix
    


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_test.append(pred)

    acc_smoothed = accuracy(preds_test, test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))

    preds_recent = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_recent.append(pred)

    acc_smoothed = accuracy(preds_recent, recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))

    # Step 4 - compare accuracy of simple classifier from tick 1

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    preds_test_simple = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment(review, lexicon)
        preds_test_simple.append(pred)

    acc_smoothed = accuracy(preds_test_simple, test_sentiments)
    print(f"Simple classifier accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test_simple, test_sentiments))

    preds_recent_simple = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment(review, lexicon)
        preds_recent_simple.append(pred)

    acc_smoothed = accuracy(preds_recent_simple, recent_sentiments)
    print(f"Simple classifier accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent_simple, recent_sentiments))

    # Significance test for NB vs simple classifer on 2016 data
    p_value = sign_test(recent_sentiments,preds_recent, preds_recent_simple)
    print(f"The p-value of the two-sided sign test for classifier_a \"{'naive bayes classifier'}\" and classifier_b \"{'simple classifier'}\": {p_value}")


if __name__ == '__main__':
    main()
