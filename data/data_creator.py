import pandas as pd
import os
import glob
from functools import *
import itertools
import tensorflow as tf
import numpy as np
import random
import re
from multiprocessing import Pool

MARKS = {'<NE>': '', '</NE>': '', '<AB>': '', '</AB>': ''}

ALL_CHAR = [
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
CHARS_MAP = {v: k for k, v in enumerate(ALL_CHAR)}
IDX_MAP = dict(list(enumerate(ALL_CHAR)))
OTHER_KEY = max(CHARS_MAP.values())
CLASS_MAP = {'B': 0, 'M': 1, 'E': 2, 'S': 3}


def word_to_idx(word):
    w_size = len(word)
    w_idx = list(map(lambda x: CHARS_MAP.get(x, OTHER_KEY), word))
    label = []
    if w_size == 1:
        label += [3]
    else:
        label = [0] + list(np.repeat([1], w_size - 2)) + [2]

    return w_idx, label


def get_feature(tokens, k):
    n_tokens = len(tokens)
    padded_tokens = tokens + [0] * k
    res = []
    for i in range(n_tokens):
        res.append(padded_tokens[i:i + k + 1])
    return res


def make_example(seq_features, labels, key):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(seq_features)
    ex.context.feature["seq_length"].int64_list.value.append(sequence_length)
    ex.context.feature["key"].int64_list.value.append(key)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["seq_feature"]
    fl_labels = ex.feature_lists.feature_list["label"]
    for feature, label in zip(seq_features, labels):
        fl_tokens.feature.add().int64_list.value.extend(feature)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex


def save_to_tfrecords(data_path, output_path, type, k):
    all_files = glob.glob(os.path.join(data_path, '*.txt'))
    random.shuffle(all_files)
    train_size = int(0.8 * len(all_files))
    train = all_files[:train_size]
    test = all_files[train_size:]

    def write(files, prefix, type):
        if not os.path.isdir(os.path.join(os.getcwd(), output_path, prefix)):
            os.makedirs(os.path.join(output_path, prefix))
        for file in files:
            words_all = []
            print(file)
            lines = open(file, 'r', encoding='utf-8')
            for line in lines:
                line = reduce(lambda a, kv: a.replace(*kv), MARKS.items(), line)
                sentence = line.split(" ")
                words = [[word for word in s.split("|") if word not in ['', '\n']] for s in sentence]
                words = filter(lambda x: len(x) > 0, words)
                words_all.extend(list(words))
            lines.close()
            word_idxs = list(map(lambda s: list(map(lambda w: word_to_idx(w), s)), words_all))
            st_idx, label = map(list, zip(
                *list(
                    map(lambda s: tuple(map(lambda x: list(itertools.chain.from_iterable(x)), list(zip(*s)))),
                        word_idxs))))
            input_feature = list(map(lambda x: get_feature(x, k), st_idx))

            # Write all examples into a TFRecords file
            f_name = re.search('([0-9].*).txt', file).group(1)

            with open(os.path.join(output_path, prefix, type + '_' + f_name + '.tf'), 'w') as fp:
                writer = tf.python_io.TFRecordWriter(fp.name)
                for key, sequence in enumerate(zip(input_feature, label)):
                    seq_input, label = sequence
                    ex = make_example(seq_input, label, key)
                    writer.write(ex.SerializeToString())
                writer.close()

    write(train, "train", type)
    write(test, "test", type)


def read_and_decode_single_example(filenames, shuffle=False, num_epochs=None):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep size down
    # filename_queue = tf.train.string_input_producer([filename], num_epochs=10)
    filename_queue = tf.train.string_input_producer(filenames,
                                                    shuffle=shuffle, num_epochs=num_epochs)

    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_ex = reader.read(filename_queue)
    context, sequences = tf.parse_single_sequence_example(serialized_ex,
                                                          context_features={
                                                              "seq_length": tf.FixedLenFeature([], dtype=tf.int64)
                                                          },
                                                          sequence_features={
                                                              "seq_feature": tf.VarLenFeature(dtype=tf.int64),
                                                              "label": tf.VarLenFeature(dtype=tf.int64)
                                                          })
    return context, sequences


def fn(cat):
    return save_to_tfrecords("_BEST/" + cat, "_tf_records_k2", cat, 2)


if __name__ == "__main__":
    category = ['article', 'encyclopedia', 'news', 'novel']
    p = Pool(4)
    p.map(fn, category)
