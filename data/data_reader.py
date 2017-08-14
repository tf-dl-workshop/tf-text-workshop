import pandas as pd
import glob
from functools import reduce
import itertools
import tensorflow as tf
import numpy as np
from tensorflow.contrib.training import batch_sequences_with_states
from tensorflow.contrib import learn
from scipy.ndimage.interpolation import shift

"""
Transform list of files to list of words,
removing new line character
and replace name entity '<NE>...</NE>' and abbreviation '<AB>...</AB>' symbol
"""

repls = {'<NE>': '', '</NE>': '', '<AB>': '', '</AB>': ''}

words_all = []
lines = open("data/_BEST/article/article_00001.txt", 'r')
for line in lines:
    line = reduce(lambda a, kv: a.replace(*kv), repls.items(), line)
    sentence = line.split(" ")
    words = map(lambda x: [word for word in x.split("|") if word not in ['', '\n']], sentence)
    words_all.extend(words)

# create map of dictionary to character
CHARS = [
    '', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
    ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    'z', '}', '~', 'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช',
    'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท',
    'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ',
    'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ฯ', 'ะ', 'ั', 'า',
    'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'ฺ', 'เ', 'แ', 'โ', 'ใ', 'ไ',
    'ๅ', 'ๆ', '็', '่', '้', '๊', '๋', '์', 'ํ', '๐', '๑', '๒', '๓',
    '๔', '๕', '๖', '๗', '๘', '๙', '‘', '’', '\ufeff', 'other'
]
CHARS_MAP = {v: k for k, v in enumerate(CHARS)}
IDX_MAP = dict(list(enumerate(CHARS)))
other_key = max(CHARS_MAP.values())
class_map = {'B': 0, 'M': 1, 'E': 2, 'S': 3}


def word_to_idx(words):
    w_size = len(words)
    w_idx = list(map(lambda x: CHARS_MAP.get(x, other_key), words))
    label = []
    if w_size == 1:
        label += [3]
    else:
        label = [0] + list(np.repeat([1], w_size)) + [2]

    return w_idx, label


def get_feature(tokens, k):
    n_tokens = len(tokens)
    padded_tokens = tokens + [0] * k
    res = []
    for i in range(n_tokens):
        res.append(padded_tokens[i:i + k + 1])
    return res


word_idxs = list(map(lambda s: list(map(lambda w: word_to_idx(w), s)), words_all))
st_idx, label = map(list, zip(
    *list(map(lambda s: tuple(map(lambda x: list(itertools.chain.from_iterable(x)), list(zip(*s)))), word_idxs))))
input_feature = list(map(lambda x: get_feature(x, 2), st_idx))


def make_example(seq_features, labels, key):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(seq_features)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    ex.context.feature["key"].int64_list.value.append(key)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["seq_feature"]
    fl_labels = ex.feature_lists.feature_list["label"]
    for feature, label in zip(seq_features, labels):
        fl_tokens.feature.add().int64_list.value.extend(feature)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex


if False:
    # Write all examples into a TFRecords file
    with open("data/_tf_records/example.tf", 'w') as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)
        for key, sequence in enumerate(zip(input_feature, label)):
            seq_input, label = sequence
            ex = make_example(seq_input, label, key)
            writer.write(ex.SerializeToString())
        writer.close()


def read_and_decode_single_example(filenames, shuffle=False, num_epochs=None):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep size down
    # filename_queue = tf.train.string_input_producer([filename], num_epochs=10)
    filename_queue = tf.train.string_input_producer(filenames,
                                                    shuffle=shuffle, num_epochs=num_epochs)

    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    key, serialized_ex = reader.read(filename_queue)
    context, sequences = tf.parse_single_sequence_example(serialized_ex,
                                                          context_features={
                                                              "seq_length": tf.FixedLenFeature([], dtype=tf.int64)
                                                          },
                                                          sequence_features={
                                                              # We know the length of both fields. If not the
                                                              # tf.VarLenFeature could be used
                                                              "seq_feature": tf.FixedLenSequenceFeature([3],
                                                                                                         dtype=tf.int64),
                                                              "label": tf.FixedLenSequenceFeature([], dtype=tf.int64)
                                                          })
    return (key, context, sequences)


key, contexts, sequences = read_and_decode_single_example(["data/_tf_records/example.tf"])

initial_state_values = tf.zeros((10,), dtype=tf.float32)
initial_states = {"lstm_state": initial_state_values}

batch = batch_sequences_with_states(
    input_key=key,
    input_sequences=sequences,
    input_context=contexts,
    input_length=tf.to_int32(contexts['length']),
    make_keys_unique=True,
    initial_states=initial_states,
    num_unroll=4,
    batch_size=10,
    num_threads=2,
    capacity=20)

words = batch.sequences["seq_feature"]
b_k = batch.key

res = learn.run_n({"words": words, "keys": b_k}, n=2, feed_dict=None)


def idx_to_words(arr):
    learn.run_feeds()
    return list(map(lambda idx: ''.join(list(map(lambda x: IDX_MAP[x], idx))), arr['words']))


list(map(lambda x: idx_to_words(x), res))
