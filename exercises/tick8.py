from utils.markov_models import load_dice_data
import os
from exercises.tick7 import estimate_hmm
import random
from math import log
import numpy as np

from typing import List, Dict, Tuple


def viterbi(observed_sequence: List[str], transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed
    sequence and a model. Use the same symbols for the start and end observations as in tick 7 ('B' for the start 
    observation and 'Z' for the end observation).

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    new_sequence = observed_sequence.copy()
    new_sequence.insert(0, 'B')
    new_sequence += 'Z'
    T = len(new_sequence)
    states = set([key[0] for key in transition_probs])
    previous_states = [] # ψ
    path_probabilities = [{}] # δ
    
    for s in states:
        b_s = emission_probs[(s, new_sequence[0])]
        try:
            delta_i = log(b_s)
        except ValueError:
            delta_i = float('-inf')
        path_probabilities[0][s] = delta_i

    for t in range(1, T):
        path_probabilities.append({})
        previous_states.append({})
        for s in states:
            b_s = emission_probs[(s, new_sequence[t])]
            delta_list = []
            for i in states:
                a_is = transition_probs[(i, s)]
                try:
                    delta_i = path_probabilities[t-1][i] + log(a_is) + log(b_s), i
                except ValueError:
                    delta_i = float('-inf'), i
                delta_list.append(delta_i)

            path_probabilities[t][s] = max(delta_list, key=lambda x: x[0])[0]
            previous_states[t-1][s] = max(delta_list, key=lambda x: x[0])[1]
    
    best_path = []
    best_state = max(path_probabilities[-1], key = path_probabilities[-1].get)
    best_path.insert(0, best_state)


    for i in range(T-2, -1, -1):
        best_state = previous_states[i][best_state]
        best_path.insert(0, best_state)
    
    # Remove artificially added start and end state
    del best_path[0]
    del best_path[-1]

    return best_path




def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    TP = 0
    FP = 0
    for i, test in enumerate(pred):
        for j, p in enumerate(test):
            if p == 1:
                if true[i][j] == 1:
                    TP += 1
                else:
                    FP += 1
    if TP + FP == 0:
        print()
    return TP / (TP + FP)


def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    TP = 0
    FN = 0
    for i, test in enumerate(pred):
        for j, p in enumerate(test):
            if true[i][j] == 1:
                if p == 1:
                    TP += 1
                else:
                    FN += 1
    return TP / (TP + FN)


def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    TP = 0
    FN = 0
    FP = 0
    for i, test in enumerate(pred):
        for j, p in enumerate(test):
            if true[i][j] == 1 and p == 1:
                TP += 1
            elif true[i][j] == 1 and p == 0:
                FN += 1
            elif p == 1:
                FP += 1
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    return 2 * P * R / (P + R)



def cross_validation_sequence_labeling(data: List[Dict[str, List[str]]]) -> Dict[str, float]:
    """
    Run 10-fold cross-validation for evaluating the HMM's prediction with Viterbi decoding. Calculate precision, recall, and F1 for each fold and return the average over the folds.

    @param data: the sequence data encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'
    @return: a dictionary with keys 'recall', 'precision', and 'f1' and its associated averaged score.
    """
    # Generate folds 
    N = 10
    copy_data = data.copy()
    list_of_folds = []
    fold_size = len(data) // N
    for _ in range(N-1):
        curr_fold = []
        for _ in range(fold_size):
            # generate a random index to take an instance from the training data
            rand_index = random.randint(0,len(copy_data)-1)
            curr_fold.append(copy_data[rand_index])
            # remove the instance from the training data
            copy_data.pop(rand_index)
        list_of_folds.append(curr_fold)
    list_of_folds.append(copy_data)

    # Cross validate 
    precisions = []
    recalls = []
    f1s = []
    for test_fold in list_of_folds:
        copy_data = list_of_folds.copy()
        copy_data.remove(test_fold)
        training_data = [sequence for fold in copy_data for sequence in fold]
        predictions = []
        true_sequence = []
        transition_probs, emission_probs = estimate_hmm(training_data)
        
        for sequence in training_data:
            predictions.append(list(map(lambda x: x == 'W', viterbi(sequence['observed'], transition_probs, emission_probs))))
            true_sequence.append(list(map(lambda x: x == 'W', sequence['hidden'])))
        
            precisions.append(precision_score(predictions, true_sequence))
            recalls.append(recall_score(predictions, true_sequence))
            f1s.append(f1_score(predictions, true_sequence))
    
    return {'precision': np.mean(precisions), 'recall': np.mean(recalls), 'f1': np.mean(f1s)}


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    dice_data_shuffled = random.sample(dice_data, len(dice_data))
    dev_size = int(len(dice_data) / 10)
    train = dice_data_shuffled[dev_size:]
    dev = dice_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm(train)

    for sample in dev_observed_sequences:
        prediction = viterbi(sample, transition_probs, emission_probs)
        predictions.append(prediction)

    predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    print(f"Evaluating HMM using cross-validation with 10 folds.")

    cv_scores = cross_validation_sequence_labeling(dice_data)

    print(f" Your cv average precision using the HMM: {cv_scores['precision']}")
    print(f" Your cv average recall using the HMM: {cv_scores['recall']}")
    print(f" Your cv average F1 using the HMM: {cv_scores['f1']}")



if __name__ == '__main__':
    main()
