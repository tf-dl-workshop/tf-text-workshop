import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import metrics
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *
from tensorflow.contrib import seq2seq
from data.data_reader import *


def rnn_segment(features, targets, mode, params):
    seq_feature = features['seq_feature']
    seq_length = features['seq_length']
    with tf.variable_scope("emb"):
        embeddings = tf.get_variable("char_emb", shape=[params['num_char'], params['emb_size']])
    seq_emb = tf.nn.embedding_lookup(embeddings, seq_feature)
    batch_size = tf.shape(seq_feature)[0]
    time_step = tf.shape(seq_feature)[1]
    flat_seq_emb = tf.reshape(seq_emb, shape=[batch_size, time_step, 3 * params['emb_size']])
    cell = rnn.GRUCell(params['rnn_units'])
    dropout_cell = rnn.DropoutWrapper(cell, params['input_keep_prob'], params['output_keep_prob'])
    projection_cell = rnn.OutputProjectionWrapper(dropout_cell, params['num_class'])
    logits, _ = tf.nn.dynamic_rnn(projection_cell, flat_seq_emb, sequence_length=seq_length, dtype=tf.float32)
    weight_mask = tf.to_float(tf.sequence_mask(seq_length))
    loss = seq2seq.sequence_loss(logits, targets, weights=weight_mask)
    train_op = layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer=tf.train.AdamOptimizer,
        clip_gradients=params['grad_clip'],
        summaries=[
            "learning_rate",
            "loss",
            "gradients",
            "gradient_norm",
        ])
    pred_classes = tf.to_int32(tf.argmax(input=logits, axis=2))
    accuracy = tf.reduce_mean(metrics.accuracy(pred_classes, targets, weights=weight_mask))
    predictions = {
        "classes": pred_classes
    }
    return learn.ModelFnOps(mode, predictions, loss, train_op, eval_metric_ops={"accuracy": accuracy})


def data_provider(batch_size):
    def input_fn():
        _, contexts, sequences = read_and_decode_single_example(["data/_tf_records/example.tf"])

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


model_params = dict(num_class=len(class_map), num_char=len(CHARS_MAP), emb_size=64, rnn_units=64, input_keep_prob=1.0,
                    output_keep_prob=1.0, learning_rate=10e-1, grad_clip=0.7)
estimator = learn.Estimator(model_fn=rnn_segment
                            , params=model_params
                            , model_dir="model/_test"
                            , config=learn.RunConfig(save_checkpoints_secs=30,
                                                     keep_checkpoint_max=2))

train_input_fn = data_provider(batch_size=32)

validation_monitor = monitors.ValidationMonitor(input_fn=train_input_fn,
                                                eval_steps=10,
                                                every_n_steps=200,
                                                name='validation')

estimator.fit(input_fn=train_input_fn, steps=10000, monitors=[validation_monitor])


# x = tf.convert_to_tensor([[1, 2, 3], [3, 5, 0]])
# y = tf.convert_to_tensor([[1, 2, 3], [3, 5, 0]])
# acc = metrics.accuracy(x, y)
# init_g = tf.global_variables_initializer()
# init_l = tf.local_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_g)
#     sess.run(init_l)
#     print(sess.run(acc))
#     print(sess.run(tf.reduce_mean(acc)))
