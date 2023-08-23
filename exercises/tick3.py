from utils.sentiment_detection import clean_plot, chart_plot, best_fit
from typing import List, Tuple, Callable
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import log, log2


def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words. Use this to
    create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    m, c = best_fit(token_frequencies_log, token_frequencies)
    print(m, c)
    return lambda rank: m*rank + c


def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:
    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of  reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    freq_dict = {}
    for subdir, dir, files in os.walk(dataset_path):
        with ThreadPoolExecutor(50) as executor:
            files = [os.path.join(dataset_path,f) for f in files]
            output_files = [executor.submit(lambda f: open(f,'r').read(), f) for f in files]
            for review in as_completed(output_files):
                review_text = review.result()
                for word in review_text.split(' '):
                    freq_dict[word] = freq_dict.setdefault(word, 0) + 1
    freq_list = [(k,v) for k,v in freq_dict.items()]
    freq_list.sort(reverse=True,key=lambda y: y[1])
    return freq_list


def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    chart_plot([(i+1, frequencies[i][1]) for i in range(10000)], '10000 most common words', 'Rank', 'Frequency')
         


def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. Plot the Task 1 words on the frequency-rank plot as a separate series on the same plot (i.e., tell the
    plotter to draw it as an additional graph on top of the graph from above function).

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    words = ['funny','exciting','good','bad','interesting','dissapointing','great','liked','hated','terrible']
    rank_freq = []
    for word in words:
        for i, tupl in enumerate(frequencies):
            if tupl[0] == word:
                rank_freq.append([i, frequencies[i][1]])
    unsorted_rank_freq = rank_freq.copy()
    rank_freq.sort(reverse=True, key=lambda y: y[0])
    chart_plot([rank_freq[i] for i in range(len(rank_freq))], '10000 most common words', 'Rank', 'Frequency')

    rank_f = [(i+1, frequencies[i][1]) for i in range(10000)]
    log_rank_f = [(log(i+1), log(frequencies[i][1])) for i in range(10000)]
    bf_fun = estimate_zipf(log_rank_f, rank_f)
    for i, word in enumerate(words):
        predicted_freq = bf_fun(log(unsorted_rank_freq[i][0]))
        print(word + ': ' + str(predicted_freq) + ', ' + str(log(unsorted_rank_freq[i][0])) + ', ' + str(log(unsorted_rank_freq[i][1])))


def draw_zipf(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    rank_freq = [(i+1, frequencies[i][1]) for i in range(10000)]
    log_rank_freq = [(log(i+1), log(frequencies[i][1])) for i in range(10000)]
    chart_plot(log_rank_freq, 'Log 10000 most common words', 'Rank', 'Frequency')
    # Plot line of best fit
    bf_fun = estimate_zipf(log_rank_freq, rank_freq)
    chart_plot([(i, bf_fun(i)) for i in range(10)], 'Log 10000 most common words', 'Rank', 'Frequency')



def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    tokens_types = []
    types = set()
    total_token_count = 0
    for subdir, dir, files in os.walk(dataset_path):
        with ThreadPoolExecutor(50) as executor:
            files = [os.path.join(dataset_path,f) for f in files]
            output_files = [executor.submit(lambda f: open(f,'r').read(), f) for f in files]
            for review in as_completed(output_files):
                review_text = review.result()
                for word in review_text.split(' '):
                    total_token_count += 1
                    types.add(word)
                    if log2(total_token_count).is_integer():
                        print(total_token_count)
                        tokens_types.append((total_token_count, len(types)))
    tokens_types.append((total_token_count, len(types)))
    return tokens_types
    


def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    chart_plot([(log(type_counts[i][1]) , log(type_counts[i][0])) for i in range(len(type_counts))], 'Log unique words against total words', 'Types', 'Tokens')


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    frequencies = count_token_frequencies(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))

    draw_frequency_ranks(frequencies)
    draw_selected_words_ranks(frequencies)

    clean_plot()
    draw_zipf(frequencies)

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    draw_heap(tokens)


if __name__ == '__main__':
    main()
