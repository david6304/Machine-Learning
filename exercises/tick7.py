from utils.markov_models import load_dice_data, print_matrices
import os
from typing import List, Dict, Tuple


def get_transition_probs(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden dice types using maximum likelihood estimation. 
    Counts the number of times each state sequence appears and divides it by the count of all transitions 
    going from that state. The table must include proability values for all state-state pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
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


def get_emission_probs(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden dice states to observed dice rolls, using maximum 
    likelihood estimation. Counts the number of times each dice roll appears for the given state (fair or loaded)
    and divides it by the count of that state. The table must include proability values for all state-observation
    pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @param observed_sequences: A list of dice roll sequences
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


def estimate_hmm(training_data: List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and
    emission probabilities. We use 'B' for the start state and 'Z' for the end state, both for emissions
    and transitions.

    @param training_data: The dice roll sequence data (visible dice rolls and hidden dice types), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs(hidden_sequences)
    emission_probs = get_emission_probs(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))
    transition_probs, emission_probs = estimate_hmm(dice_data)
    print(f"The transition probabilities of the HMM:")
    print_matrices(transition_probs)
    print(f"The emission probabilities of the HMM:")
    print_matrices(emission_probs)

if __name__ == '__main__':
    main()
