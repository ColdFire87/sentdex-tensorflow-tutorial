import numpy as np
import random
import json
from scipy.sparse import csr_matrix, save_npz
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
num_lines = 10000000  # 10 Mil


def create_lexicon(pos, neg):
    print('Creating lexicon...')

    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            contents = f.readlines()
            for line in contents[:num_lines]:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)

    return l2


def sample_handling(sample, lexicon, classification):
    print('Handling sample `{}`...'.format(sample))

    feature_set = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for line in contents[:num_lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            feature_set.append([features, classification])

    return feature_set


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    _lexicon = create_lexicon(pos, neg)

    features = []
    features += sample_handling(pos, _lexicon, [1, 0])
    features += sample_handling(neg, _lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size * len(features))
    _x_train = list(features[:, 0][:-testing_size])
    _y_train = list(features[:, 1][:-testing_size])
    _x_test = list(features[:, 0][-testing_size:])
    _y_test = list(features[:, 1][-testing_size:])

    return _lexicon, _x_train, _y_train, _x_test, _y_test


if __name__ == '__main__':
    # Create lists
    lexicon, x_train, y_train, x_test, y_test = create_feature_sets_and_labels('small_data/data/pos.txt',
                                                                               'small_data/data/neg.txt')

    # Convert lists to sparse matrices
    [x_train, y_train, x_test, y_test] = list(map(csr_matrix, [x_train, y_train, x_test, y_test]))

    print('Saving processed data...')

    # Save lexicon
    with open('small_data/saved/lexicon.json', 'w', encoding='latin-1') as f:
        json.dump(lexicon, f)

    # Save sparse matrices
    save_npz('small_data/saved/x_train.npz', x_train)
    save_npz('small_data/saved/y_train.npz', y_train)
    save_npz('small_data/saved/x_test.npz', x_test)
    save_npz('small_data/saved/y_test.npz', y_test)

    print('Done!')
