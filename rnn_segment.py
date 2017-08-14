import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from tensorflow.contrib import seq2seq


def rnn_segment(features, targets, mode, params):
    seq_feature = features['seq_feature']
    seq_length = features['seq_length']
    with tf.variable_scope("emb"):
        embeddings = tf.get_variable("char_emb", shape=[params['char_size'], params['emb_size']])
    seq_emb = tf.nn.embedding_lookup(embeddings, seq_feature)
    cell = rnn.GRUCell(params['rnn_units'])
    rnn_output, _ = tf.nn.dynamic_rnn(cell, seq_emb, sequence_length=seq_length, dtype=tf.float32)
    mask_weights = tf.to_float(tf.sequence_mask(seq_length))
    loss = seq2seq.sequence_loss(rnn_output, targets, weights=mask_weights)
    train_op = layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer=tf.train.AdamOptimizer,
        summaries=[
            "learning_rate",
            "loss",
            "gradients",
            "gradient_norm",
        ])
    pred = tf.argmax(rnn_output, axis=1)

    return learn.ModelFnOps(mode, pred, loss, train_op)
