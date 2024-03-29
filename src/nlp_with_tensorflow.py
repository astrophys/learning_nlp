# Author : Ali Snedden
# Date   : 02/24/24
# License:
#   a) Google code - Apache 2.0 License (per webpage)
#   #) Ali code - GPL-3 (compatable w/ Apache 2.0)
# Goals (ranked by priority) :
#
# Refs :
#   a)
#   #) https://www.nltk.org/book/ch06.html
#
import os
import sys
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def load_imdb_sentiment_analysis_dataset(path: str, seed: int =123):
    """Loads the IMDb movie reviews sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 25000
        Number of test samples: 25000
        Number of categories: 2 (0 - negative, 1 - positive)

    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015

        Download and uncompress archive from:
        http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    """
    # I basically just copied and pasted this without even reading it.
    imdb_data_path = os.path.join(path)

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))


def get_num_words_per_sample(sample_texts):
    """Returns the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()


def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 2),
                                          num_ngrams=50):
    """Plots the frequency distribution of n-grams.

    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, mplt), The range of n-gram values to consider.
            Min and mplt are the lower and upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
            'ngram_range': (1, 1),
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # idxices). This also converts every text to an array the length of
    # vocabulary, where every element idxicates the count of the n-gram
    # corresponding at that idxex in vocabulary.
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    # ali changed get_feature_names() is obsolete
    all_ngrams = list(vectorizer.get_feature_names_out())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()


def main():
    """Loads the IMDb movie reviews sentiment analysis dataset.

    # Arguments
        N/A
    # Returns

    # References
    """
    parser = argparse.ArgumentParser(
                    description="Do age analysis. fileL, sheetL and "
                                "generation should all be in order w/r/t each other")
    parser.add_argument('--imbdpath', metavar='path/to/imbd_database/', type=str,
                        help='Path to the IMBD database')
    args = parser.parse_args()
    imbddir = args.imbdpath
    # train = (content_list, class_list)
    # test = (content_list, class_list)
    (train, test) = load_imdb_sentiment_analysis_dataset(imbddir, 123)

    ###### Important Metrics / Sanity check ######
    # Number of samples and classes per sample
    print("number of training : {}".format(len(train[0])))
    print("     class 0 = {}".format(np.sum(train[1] == 0)))
    print("     class 1 = {}".format(np.sum(train[1] == 1)))
    print("number of testing  : {}".format(len(test[0])))
    print("     class 0 = {}".format(np.sum(test[1] == 0)))
    print("     class 1 = {}".format(np.sum(test[1] == 1)))

    # Number of samples per class
    # Number of words per sample
    # Freq distribution
    plot_sample_length_distribution(train[0])
    print("num words per sample : {}".format(get_num_words_per_sample(train[0])))

    plot_frequency_distribution_of_ngrams(train[0],
                                          ngram_range=(1, 2),
                                          num_ngrams=50)
    sys.stdout.flush()

    sys.exit(0)


if __name__ == "__main__":
    main()
