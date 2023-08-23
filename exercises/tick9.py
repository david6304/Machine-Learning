from utils.markov_models import load_bio_data
import os
import random
from exercises.tick8 import recall_score, precision_score, f1_score

from typing import List, Dict, Tuple
from math import log


def get_transition_probs_bio(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden feature types using maximum likelihood estimation.

    @param hidden_sequences: A list of feature sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    transitions_probs = {}
    states = set()
    total_transistions = {}
    for sequence in hidden_sequences:
        # Get all states 
        states.add(sequence[-1])
        for i in range(len(sequence)-1):
            states.add(sequence[i])
            # Count total transitions as anytime that state appears
            if sequence[i] in total_transistions:
                total_transistions[sequence[i]] += 1
            else:
                total_transistions[sequence[i]] = 1
            # Count specific transitions
            transition = (sequence[i], sequence[i+1])
            if transition in transitions_probs:
                transitions_probs[transition] += 1
            else:
                transitions_probs[transition] = 1
    
    # Add any transitions that had 0 occurences
    for state1 in states:
        for state2 in states:
            if not (state1, state2) in transitions_probs:
                transitions_probs[(state1, state2)] = 0
    
    # Divide all transitions by total to get probability
    for transition in transitions_probs:
        if transition[0] in total_transistions:
            transitions_probs[transition] /= total_transistions[transition[0]]
    

    return transitions_probs 


def get_emission_probs_bio(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden feature states to visible amino acids, using maximum likelihood estimation.
    @param hidden_sequences: A list of feature sequences
    @param observed_sequences: A list of amino acids sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    emission_probs = {}
    observations = set([])
    states = set()
    total_states = {}
    for i, sequence in enumerate(observed_sequences):
        # Get all observations and states
        observations.add(sequence[-1])
        states.add(hidden_sequences[i][-1])
        for j in range(len(sequence)):
            observations.add(sequence[j])
            states.add(hidden_sequences[i][j])
            # Store total number of each observation
            if hidden_sequences[i][j] in total_states:
                total_states[hidden_sequences[i][j]] += 1
            else:
                total_states[hidden_sequences[i][j]] = 1
            # Store number of classes given observation
            state_given_observation = (hidden_sequences[i][j], sequence[j])
            if state_given_observation in emission_probs:
                emission_probs[state_given_observation] += 1
            else:
                emission_probs[state_given_observation] = 1
    
    # Add any transitions that had 0 occurences
    for observation in observations:
        for state in states:
            if not (state, observation) in emission_probs:
                emission_probs[(state, observation)] = 0
    
    # Divide all transitions by total to get probability
    for observation in emission_probs:
        if observation[0] in total_states:
            emission_probs[observation] /= total_states[observation[0]]

    return emission_probs


def estimate_hmm_bio(training_data:List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities.

    @param training_data: The biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs_bio(hidden_sequences)
    emission_probs = get_emission_probs_bio(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def viterbi_bio(observed_sequence, transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model.

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




def self_training_hmm(training_data: List[Dict[str, List[str]]], dev_data: List[Dict[str, List[str]]],
    unlabeled_data: List[List[str]], num_iterations: int) -> List[Dict[str, float]]:
    """
    The self-learning algorithm for your HMM for a given number of iterations, using a training, development, 
    and unlabeled dataset (no cross-validation to be used here since only very limited computational resources are available.)

    @param training_data: The training set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param dev_data: The development set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param unlabeled_data: Unlabeled sequence data of amino acids, encoded as a list of sequences.
    @num_iterations: The number of iterations of the self_training algorithm, with the initial HMM being the first iteration.
    @return: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    """

    return_dict = []
    psuedo_labelled_data = []
    
    for _ in range(num_iterations+1):
        dev_observed_sequences = [x['observed'] for x in dev_data]
        dev_hidden_sequences = [x['hidden'] for x in dev_data]
        predictions = []
        transition_probs, emission_probs = estimate_hmm_bio(training_data + psuedo_labelled_data)
        psuedo_labelled_data = []

        for sample in dev_observed_sequences:
            prediction = viterbi_bio(sample, transition_probs, emission_probs)
            predictions.append(prediction)
        predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
        dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]

        return_dict += [{'precision': precision_score(predictions_binarized, dev_hidden_sequences_binarized), 
                    'recall': recall_score(predictions_binarized, dev_hidden_sequences_binarized),
                    'f1': f1_score(predictions_binarized, dev_hidden_sequences_binarized)}]


        for sequence in unlabeled_data:
            pred = viterbi_bio(sequence, transition_probs, emission_probs)
            psuedo_labelled_data.append({'observed': sequence, 'hidden': pred})

    print(return_dict)

    return return_dict

        

def visualize_scores(score_list:List[Dict[str,float]]) -> None:
    """
    Visualize scores of the self-learning algorithm by plotting iteration vs scores.

    @param score_list: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    @return: The most likely single sequence of hidden states
    """
    from utils.sentiment_detection import clean_plot, chart_plot

    chart_plot([(i, score_list[i]['precision']) for i in range(len(score_list))], 'Performance of self training HMM', 'Iteration', 'Score')
    chart_plot([(i, score_list[i]['recall']) for i in range(len(score_list))], 'Performance of self training HMM', 'Iteration', 'Score')
    chart_plot([(i, score_list[i]['f1']) for i in range(len(score_list))], 'Performance of self training HMM', 'Iteration', 'Score')


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    bio_data = load_bio_data(os.path.join('data', 'markov_models', 'bio_dataset.txt'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    bio_data_shuffled = random.sample(bio_data, len(bio_data))
    dev_size = int(len(bio_data_shuffled) / 10)
    train = bio_data_shuffled[dev_size:]
    dev = bio_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm_bio(train)

    for sample in dev_observed_sequences:
        prediction = viterbi_bio(sample, transition_probs, emission_probs)
        predictions.append(prediction)
    predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    unlabeled_data = []
    with open(os.path.join('data', 'markov_models', 'bio_dataset_unlabeled.txt'), encoding='utf-8') as f:
        content = f.readlines()
        for i in range(0, len(content), 2):
            unlabeled_data.append(list(content[i].strip())[1:])

    scores_each_iteration = self_training_hmm(train, dev, unlabeled_data, 5)

    visualize_scores(scores_each_iteration)

# irs38

if __name__ == '__main__':
    main()