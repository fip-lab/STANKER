# encoding=utf-8
import csv
import os
import tensorflow as tf
from collections import deque


flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_bool("train", None, "Whether do training.")

flags.DEFINE_bool("test", None, "Whether do testing.")

flags.DEFINE_integer("input_size", None, "The size of input data.")

flags.DEFINE_integer("num_units", None, "The hidden units of fc layer.")

flags.DEFINE_integer("batch_size", None, "Total number of data each batch.")

flags.DEFINE_integer("epoches", None, "Total number of training epochs to perform.")

flags.DEFINE_float("learning_rate", None, "Learning rate for optimizer.")

# Default parameters
flags.DEFINE_integer("num_labels", 2, "The number of labels.")

flags.DEFINE_string("training_file", "../data/train.csv", "The file for training.")

flags.DEFINE_string("test_file", "../data/test.csv", "The file for testing.")

flags.DEFINE_string("checkpoint_path", "../output/model", "The checkpoint file.")

flags.DEFINE_string("result_file", "../output/test_results.tsv", "The csv file for saving the test results")


def create_initializer(initializer_range=0.2):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def judge(inputs, units):

    layer1_outputs = tf.layers.dense(inputs, 2, activation=None, kernel_initializer=create_initializer())
    return layer1_outputs


def get_batches(file, batch_size):
    all_data = deque([])
    all_labels = deque([])
    with open(file, "r", encoding="utf-8") as fp:
        csv_reader = csv.reader(fp)
        for item in csv_reader:
            data = []
            for num in item[:-1]:
                data.append(float(num))
            all_data.append(data)
            all_labels.append(int(item[-1]))
            
    assert (len(all_data) == len(all_labels))
    
    data_labels_batches = []
    while len(all_data) > batch_size:
        data_batch = []
        labels_batch = []
        for i in range(batch_size):
            data_batch.append(all_data.popleft())
            labels_batch.append(all_labels.popleft())
        data_labels_batches.append([data_batch, labels_batch])

    content_batch = []
    labels_batch = []
    while all_data:
        content_batch.append(all_data.popleft())
        labels_batch.append(all_labels.popleft())
    data_labels_batches.append([content_batch, labels_batch])

    return data_labels_batches


def output_results(file, results):
    csv.register_dialect('tsv', delimiter='\t')
    with open(file, "w", encoding="utf-8", newline="") as fp:
        csv_writer = csv.writer(fp, 'tsv')
        for item in results:
            csv_writer.writerow(list(item))


def main(_):
    train = FLAGS.train
    test = FLAGS.test
    input_size = FLAGS.input_size
    num_units = FLAGS.num_units
    batch_size = FLAGS.batch_size
    epoches = FLAGS.epoches
    learning_rate = FLAGS.learning_rate
    
    num_labels = FLAGS.num_labels 
    training_file = FLAGS.training_file
    test_file = FLAGS.test_file
    checkpoint_path = FLAGS.checkpoint_path
    result_file = FLAGS.result_file
    
    # fc scope
    inputs = tf.placeholder(tf.float32, shape=[None, input_size])
    logits = judge(inputs, num_units)
    
    # loss scope
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    
    labels = tf.placeholder(tf.int32, shape=[None])
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    
    # train
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if train:
            if not os.path.exists("../output"):
                os.mkdir("../output")
            data_labels_batches = get_batches(training_file, batch_size)
            for i in range(epoches):
                for data_lables_batch in data_labels_batches:
                    data_batch = data_lables_batch[0]
                    labels_batch = data_lables_batch[1]
                     
                    sess.run(train_op, feed_dict={inputs: data_batch, labels: labels_batch})
            saver = tf.train.Saver()
            saver.save(sess, checkpoint_path)
        
        if test:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            results = []
            data_labels_batches = get_batches(test_file, batch_size)
            for data_labels_batch in data_labels_batches:
                data_batch = data_labels_batch[0]
                batch_results = sess.run(probabilities, feed_dict={inputs: data_batch})
                results.extend(batch_results)
            output_results(result_file, results)
            
            
if __name__ == "__main__":
    flags.mark_flag_as_required("train")
    flags.mark_flag_as_required("test")
    flags.mark_flag_as_required("input_size")
    flags.mark_flag_as_required("num_units")
    flags.mark_flag_as_required("batch_size")
    flags.mark_flag_as_required("epoches")
    flags.mark_flag_as_required("learning_rate")
    tf.app.run()
