# encoding=utf-8
import os
import tensorflow as tf
from file_io import *

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data_a_file", None, "file a.")

flags.DEFINE_string("data_b_file", None, "file b.")

flags.DEFINE_string("output_file", None, "output file.")


def concatenate(vector_a, vector_b):
    assert len(vector_a) == len(vector_b)
    output_vector = []
    output_vector.extend(vector_a)
    output_vector.extend(vector_b)
    return output_vector


def merge(data_a_file, data_b_file, output_file):
    data_a = read_csv_file(data_a_file)
    data_b = read_csv_file(data_b_file)
    assert len(data_a) == len(data_b)
    output = []
    for i in range(len(data_a)):
        temp = []
        item_a = data_a[i]
        item_b = data_b[i]
        assert item_a[0] == item_b[0]
        assert item_a[1] == item_b[1]

        label = item_a[0]

        mid = item_a[1]

        embedding_a = item_a[2:]
        embedding_b = item_b[2:]
        assert len(embedding_a) == 768
        assert len(embedding_b) == 768

        embedding = concatenate(embedding_a, embedding_b)

        temp.extend(embedding)
        temp.append(label)

        output.append(temp)
    write_csv_file(output_file, output)


def main(_):
    merge(FLAGS.data_a_file, FLAGS.data_b_file, FLAGS.output_file)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_a_file")
    flags.mark_flag_as_required("data_b_file")
    flags.mark_flag_as_required("output_file")
    tf.app.run()








