from data.data_creator import *


def predict_input_fn(sentenses, seq_length, k):
    def input_fn():
        max_length = max(seq_length)
        sentenses_idx = [list(map(lambda x: CHARS_MAP.get(x, OTHER_KEY), sentense)) for sentense in sentenses]
        pad_sentense = [s + [0] * (max_length - l) for s, l in zip(sentenses_idx, seq_length)]
        seq_feature = list(map(lambda x: get_feature(x, k), pad_sentense))
        features = {"seq_feature": tf.convert_to_tensor(seq_feature), 'seq_length': tf.convert_to_tensor(seq_length)}

        return features

    return input_fn


def insert_pipe(s, c, l):
    begin_index = np.where(c[:l] == 0)
    return ''.join(np.insert(np.array(list(s[:l])), begin_index[0], ['|'] * len(begin_index)))


def tudkum(text, estimator, k):
    text = text.replace('\n', ' ')
    sentenses = text.split(" ")
    sentenses = list(filter(lambda x: len(x) > 0, sentenses))
    seq_length = [len(sentense) for sentense in sentenses]
    classes = [x['classes'] for x in estimator.predict(input_fn=predict_input_fn(sentenses, seq_length, k))]
    sentenses = [insert_pipe(s, c, l) for s, c, l in zip(sentenses, classes, seq_length)]
    return ''.join(sentenses).split("|")[1:]

def data_provider(data_path, batch_size):
    def input_fn():
        filenames = glob.glob(os.path.join(data_path, '*.tf'))

        contexts, sequences = read_and_decode_single_example(filenames, shuffle=True)

        tensors = {**contexts, **sequences}

        batch = tf.train.batch(
            tensors=tensors,
            batch_size=batch_size,
            dynamic_pad=True,
            name="seq_batch"
        )

        for key in sequences.keys():
            batch[key] = tf.to_int32(tf.sparse_tensor_to_dense(batch[key]))

        label = tf.squeeze(batch.pop('label'), axis=2)

        return batch, label

    return input_fn