import os
import json
import numpy as np
import pandas as pd
import tqdm

from scipy.sparse import csr_matrix, save_npz, vstack

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def init_process(f_in, f_out):
    print('Pre-processing raw data. `{}` -> `{}`'.format(f_in, f_out))

    outfile = open(f_out, 'w', encoding='latin-1')
    with open(f_in, encoding='latin-1') as f:
        for line in f:
            line = line.replace('"', '')
            columns = line.split(',')
            initial_polarity = columns[0]

            if initial_polarity == '4':
                initial_polarity = [1, 0]  # positive sentiment
            elif initial_polarity == '0':
                initial_polarity = [0, 1]  # negative sentiment

            else:
                continue  # exclude neutral sentiment tweets

            tweet = columns[-1]
            outfile.write('{}:::{}'.format(initial_polarity, tweet))

    outfile.close()


def create_lexicon(f_in, f_out, f_details, granularity=2500):
    print('Creating lexicon...')

    content = ''
    with open(f_in, 'r', encoding='latin-1') as f:
        counter = 1

        for line in f:
            counter += 1

            if counter % granularity == 0:
                columns = line.split(':::')
                tweet = ' '.join(columns[1:])  # account for potential multiple ':::'
                content += ' ' + tweet

    words = word_tokenize(content)
    words = [lemmatizer.lemmatize(i) for i in words]

    lexicon = list(set(words))
    size = len(lexicon)

    with open(f_out.format(granularity, size), 'w', encoding='latin-1') as f:
        json.dump(lexicon, f)

    with open(f_details, 'w', encoding='latin-1') as f:
        f.write('{},{}'.format(granularity, size))

    return granularity, size


def convert_to_vec(f_in, f_out, lexicon_json, _lexicon_granularity, _lexicon_size, batch_size=10000):
    print('Creating bag of words...')

    # Load lexicon
    with open(lexicon_json.format(_lexicon_granularity, _lexicon_size), 'r', encoding='latin-1') as f:
        lexicon = json.load(f)

    features = None
    labels = None

    with tqdm.tqdm(total=os.path.getsize(f_in), unit='b', unit_scale=True,
                   desc='counter: {0:>7}'.format(0)) as pbar:
        with open(f_in, 'r', encoding='latin-1') as f:
            counter = 0
            features_batch = []
            labels_batch = []

            for line in f:
                counter += 1

                # Account for potential multiple ':::'
                columns = line.split(':::')
                label = columns[0]
                tweet = ' '.join(columns[1:])

                # Tokenize & lemmatize tweet
                current_words = word_tokenize(tweet)
                current_words = [lemmatizer.lemmatize(i) for i in current_words]

                # Extract current features from processed tweet based on lexicon
                current_features = np.zeros(len(lexicon))
                for word in current_words:
                    if word in lexicon:
                        current_features[lexicon.index(word)] += 1
                current_features = list(current_features)

                # Add current_features & label to batches
                features_batch.append(current_features)
                labels_batch.append(eval(label))

                # Every batch_size lines - convert batches into sparse matrices and add them to features & labels
                if counter % batch_size == 0:
                    pbar.set_description('counter: {0:>7}'.format(counter))

                    # Convert batches to sparse matrices
                    sparse_features_batch = csr_matrix(features_batch)
                    sparse_labels_batch = csr_matrix(labels_batch)

                    if features is None:
                        # First batch
                        features = sparse_features_batch
                        labels = sparse_labels_batch
                    else:
                        # Add batch to sparse matrices
                        features = vstack([features, sparse_features_batch])
                        labels = vstack([labels, sparse_labels_batch])

                    # Empty buffers
                    features_batch = []
                    labels_batch = []

                # Update progress
                pbar.update(len(line))

    # Save sparse matrices to files
    save_npz(f_out.format(_lexicon_granularity, _lexicon_size, 'features'), features)
    save_npz(f_out.format(_lexicon_granularity, _lexicon_size, 'labels'), labels)


def shuffle_data(f_in, f_out):
    print('Shuffling data `{}` -> `{}`'.format(f_in, f_out))

    df = pd.read_csv(f_in, encoding='latin-1')
    df = df.iloc[np.random.permutation(len(df))]
    df.to_csv(f_out, index=False)


if __name__ == '__main__':
    # Strip unnecessary columns (keep only tweet & label)
    init_process('large_data/data/training.1600000.processed.noemoticon.csv', 'large_data/temp/train_set.csv')

    # Shuffle dataset
    shuffle_data('large_data/temp/train_set.csv', 'large_data/temp/train_set_shuffled.csv')

    # Create lexicon from shuffled, filtered dataset (only every 2500th tweet is used)
    lexicon_granularity, lexicon_size = create_lexicon('large_data/temp/train_set_shuffled.csv',
                                                       'large_data/saved/lexicon-{}-{}.json',
                                                       'large_data/saved/lexicon-details.csv',)

    # Using the shuffled, filtered dataset and the lexicon,
    # create the feature matrix (bag of words) & labels (as sparse matrices for space efficiency)
    convert_to_vec('large_data/temp/train_set_shuffled.csv',
                   'large_data/saved/processed-train-set-{}-{}-{}.npz',
                   'large_data/saved/lexicon-{}-{}.json',
                   lexicon_granularity, lexicon_size)
