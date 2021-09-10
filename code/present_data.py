# encoding=utf-8
import os
import tensorflow as tf
from file_io import *
from bert_represent import BertRepresent

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data_file", None, "file to be processed.")

flags.DEFINE_string("output_file", None, "generated file.")

flags.DEFINE_string("model_path", None, "the model path.")


def main(_):

    data = read_csv_file(FLAGS.data_file)
    
    sess = tf.Session()
    my_bert_represent = BertRepresent(
        sess=sess,
        config_file='../bert/bert_config.json',
        vocab_file='../bert/vocab.txt',
        max_seq_length=512,
        init_checkpoint=tf.train.latest_checkpoint(FLAGS.model_path))
    sess.run(tf.global_variables_initializer())

    data_representation = []
    for item in data:
        mid = item[0]
        text = item[1]
        label = item[2]
        representation = my_bert_represent.get_sentences_representation([text])
        temp = []
        temp.append(label)
        temp.append(mid)
        temp.extend(representation[0])
        data_representation.append(temp)
    
    write_csv_file(FLAGS.output_file, data_representation)
 
 
if __name__ == "__main__":
    flags.mark_flag_as_required("data_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("model_path")
    tf.app.run()