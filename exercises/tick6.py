import os
from typing import List, Dict, Union

from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table
from math import log
from numpy import argmax

from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy


def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    pos = 0
    neg = 0
    neutral = 0
    for data in training_data:
        if data['sentiment'] == 1:
            pos += 1
        elif data['sentiment'] == 0:
            neutral += 1
        else:
            neg += 1
    return {1: log(pos/len(training_data)), -1: log(neg/len(training_data)), 0: log(neutral/len(training_data))}


def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    word_probability = {1: {}, -1: {}, 0: {}}
    pos = 0
    neg = 0
    neutral = 0
    for data in training_data:
        if data['sentiment'] == 1:
            for word in data['text']:
                if word in word_probability[1]:
                    word_probability[1][word] += 1
                else:
                    word_probability[1][word] = 1
                pos += 1
        elif data['sentiment'] == 0:
            for word in data['text']:
                if word in word_probability[0]:
                    word_probability[0][word] += 1
                else:
                    word_probability[0][word] = 1
                neutral += 1
        else:
            for word in data['text']:
                if word in word_probability[-1]:
                    word_probability[-1][word] += 1
                else:
                    word_probability[-1][word] = 1
                neg += 1

    for word in word_probability[1]:
        if word not in word_probability[-1]:
            word_probability[-1][word] = 1
        if word not in word_probability[0]:
            word_probability[0][word] = 1

    for word in word_probability[-1]:
        if word not in word_probability[1]:
            word_probability[1][word] = 1
        if word not in word_probability[0]:
            word_probability[0][word] = 1
    
    for word in word_probability[0]:
        if word not in word_probability[1]:
            word_probability[1][word] = 1
        if word not in word_probability[-1]:
            word_probability[-1][word] = 1

    vocabulary = len(word_probability[1])

    for word in word_probability[1]:
        word_probability[1][word] = log(word_probability[1][word]/(pos+vocabulary))
        word_probability[-1][word] = log(word_probability[-1][word]/(neg+vocabulary))
        word_probability[0][word] = log(word_probability[0][word]/(neutral+vocabulary))
        
    return word_probability


def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    correct = 0
    for i in range(len(pred)):
        if pred[i] == true[i]:
            correct += 1
    return correct/len(pred)


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 0, 1] for the given review
    """
    pos_sum = 0
    neg_sum = 0
    neutral_sum = 0
    for word in review:
        if word in log_probabilities[1]:
            pos_sum += log_probabilities[1][word]
        if word in log_probabilities[-1]:
            neg_sum += log_probabilities[-1][word]
        if word in log_probabilities[0]:
            neutral_sum += log_probabilities[0][word]

    maximum = argmax([class_log_probabilities[1] + pos_sum, class_log_probabilities[-1] + neg_sum, class_log_probabilities[0] + neutral_sum])
    if maximum == 0:
        return 1
    elif maximum == 1:
        return -1
    else:
        return 0


def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    N = len(agreement_table)
    print(agreement_table)
    k = 0
    P_A = 0
    P_E = 0
    vals = []
    for i, review in enumerate(agreement_table):
        if i == 0:
            k = sum(agreement_table[review][sentiment] for sentiment in agreement_table[review])
    
    for i, review in enumerate(agreement_table):
        row = []
        for j, sentiment in enumerate(agreement_table[review]):
            n_ij = agreement_table[review][sentiment]
            P_A += 1 / (k*(k-1)) * n_ij * (n_ij - 1)
            row.append(n_ij)
        vals.append(row)
    print(vals)
    for j in range(len(vals[0])):
        curr = 0
        for i in range(len(vals)):
            curr += vals[i][j]
        P_E += (curr / (N*k))**2
    

    P_A /= N
    print(P_A)
    print(P_E)
    print(k)
    return (P_A - P_E) / (1 - P_E)


def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    agreement_table = {}
    for prediction in review_predictions:
        for id, sentiment in prediction.items():
            if id == 4:
                continue
            if id not in agreement_table:
                agreement_table[id] = {-1:0, 0:0, 1:0}
            agreement_table[id][sentiment] += 1

    return agreement_table


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i+1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2022.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2022.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The cohen kappa score for the review predictions from 2019 to 2022 is {fleiss_kappa}.")



if __name__ == '__main__':
    main()
