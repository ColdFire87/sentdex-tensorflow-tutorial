## Tensorflow Neural Networks

Based on [sentdex's](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)
tutorial at [https://www.youtube.com/watch?v=6rDWwL6irG0](https://www.youtube.com/watch?v=6rDWwL6irG0)

Contains code for a DNN and an LSTM (RNN).

__NOTE: LSTM (RNN) code is buggy.__

### Prerequisites:
Required dependencies:
- Anaconda python distribution:
    - numpy
    - scipy
    - pandas
    - matplotlib
    - NLTK
- tensorflow (ideally with GPU support)
- [tqdm](https://github.com/tqdm/tqdm) (`pip install tqdm`)

Make sure you have the following folder structure _(create folders if needed)_:
```
<project_dir>
    |
    |--- preprocessing
                |
                |--- large_data
                |       |
                |       |--- data
                |       |--- saved
                |       |--- temp
                |
                |--- small_data
                        |
                        |--- data
                        |--- saved
     
```

Only the small dataset is included in the repo.

Download large dataset _(see bottom of this readme for link)_ and place it at
`preprocessing/large_data/data/training.1600000.processed.noemoticon.csv`

### Usage:

#### To preprocess small dataset:
```python
python3 preprocessing/create_sentiment_feature_sets_small.py
```
This takes as input the data in `preprocessing/small_data/data` and saves output
to `preprocessing/small_data/saved`

#### To preprocess large dataset:
```python
python3 preprocessing/create_sentiment_feature_sets_large.py
```
This takes as input the data in `preprocessing/large_data/data` and saves output
to `preprocessing/large_data/saved`. Temporary files are stored in `preprocessing/large_data/temp`.

#### To train neural network:
```python
python3 tf_nn.py
```
Use the following options in the `__main__` function to specify behaviour:
```python
SMALL_DATA = False
USE_MNIST = False

# TODO: LSTM (RNN) model is buggy (chunking part during training)
network_model = MODELS[0]  # (0 - DNN, 1 - LSTM (RNN))
```

### Improvements:

1
---
The original tutorial uses regular lists for storing large sparse matrices and
serializes them using the `pickle` module. This leads to very large files.

In one case, a generated CSV file is __19.6GB__ in size.

By transforming these lists into `scipy` sparse matrices and serializing them
as `zipped numpy arrays` we reduce the size greatly.

In the previous example, the size is reduced from __19.6GB__ to __26.7MB__.

Using sparse matrices also means we can fit all data in RAM!

2
---
Added progress bars (using `tqdm`) for:
- generating bag of words (word vectors) for large dataset.
- training epochs




### Datasets from:
- small dataset (`pos.txt` and `neg.txt`):
[https://pythonprogramming.net/static/downloads/machine-learning-data/](https://pythonprogramming.net/static/downloads/machine-learning-data/)
- large dataset: download from [http://help.sentiment140.com/for-students](http://help.sentiment140.com/for-students)


