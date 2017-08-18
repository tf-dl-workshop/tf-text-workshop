import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import metrics
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from tensorflow.contrib.learn import *
from tensorflow.contrib import seq2seq
from data.data_creator import *
from utils import *

tf.logging.set_verbosity(tf.logging.INFO)

print(tf.__version__)


def rnn_segment(features, targets, mode, params):
    seq_feature = features['seq_feature']
    seq_length = features['seq_length']
    with tf.variable_scope("emb"):
        embeddings = tf.get_variable("char_emb", shape=[params['num_char'], params['emb_size']])
    seq_emb = tf.nn.embedding_lookup(embeddings, seq_feature)
    batch_size = tf.shape(seq_feature)[0]
    time_step = tf.shape(seq_feature)[1]
    flat_seq_emb = tf.reshape(seq_emb, shape=[batch_size, time_step, (params['k'] + 1) * params['emb_size']])
    cell = rnn.LSTMCell(params['rnn_units'])
    if mode == ModeKeys.TRAIN:
        cell = rnn.DropoutWrapper(cell, params['input_keep_prob'], params['output_keep_prob'])
    projection_cell = rnn.OutputProjectionWrapper(cell, params['num_class'])
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
    pred_words = tf.logical_or(tf.equal(pred_classes, 0), tf.equal(pred_classes, 3))
    target_words = tf.logical_or(tf.equal(targets, 0), tf.equal(targets, 3))
    precision = metrics.streaming_precision(pred_words, target_words, weights=weight_mask)
    recall = metrics.streaming_recall(pred_words, target_words, weights=weight_mask)
    predictions = {
        "classes": pred_classes
    }
    eval_metric_ops = {
        "precision": precision,
        "recall": recall
    }
    return learn.ModelFnOps(mode, predictions, loss, train_op, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    model_params = dict(num_class=len(CLASS_MAP), num_char=len(CHARS_MAP), emb_size=128, rnn_units=256,
                        input_keep_prob=0.85, output_keep_prob=0.85, learning_rate=10e-4, grad_clip=1.0, k=2)
    rnn_model = learn.Estimator(model_fn=rnn_segment
                                , params=model_params
                                , model_dir="model/_rnn_model"
                                , config=learn.RunConfig(save_checkpoints_secs=30,
                                                         keep_checkpoint_max=2))

    train_input_fn = data_provider("data/_tf_records/train", batch_size=128)
    test_input_fn = data_provider("data/_tf_records/test", batch_size=512)

    validation_monitor = monitors.ValidationMonitor(input_fn=test_input_fn,
                                                    eval_steps=10,
                                                    every_n_steps=500,
                                                    name='validation')

    # rnn_model.fit(input_fn=train_input_fn, steps=1, monitors=[validation_monitor])
    rnn_model.evaluate(input_fn=test_input_fn, steps=10)

    text = """นางกุหลาบขายกุหลาบจำหน่ายไม้ดอกไม้ประดับ"""
    tudkum(text, rnn_model, model_params['k'])
