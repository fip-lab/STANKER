# encoding=utf-8
import os
import tensorflow as tf
from file_io import *
from parse_results import get_results

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("iteration", None, "training times.")


def empty_dir(dir_path):
    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        os.remove(file_path)
        

def delete_dir(dir_path):
    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        os.remove(file_path)
    os.rmdir(dir_path)


def main(_):
        iteration = FLAGS.iteration
        highest_accuracy = 0
        saved_T_prec = 0
        saved_T_rec = 0
        saved_T_F1 = 0
        saved_F_prec = 0
        saved_F_rec = 0
        saved_F_F1 = 0
        for i in range(iteration):
            os.system("python3 fc.py --train=true --test=false --input_size=1536 --num_units=128 --batch_size=8 --epoches=4 --learning_rate=0.0001")
            os.system("python3 fc.py --train=false --test=true --input_size=1536 --num_units=128 --batch_size=8 --epoches=4 --learning_rate=0.0001")
            accuracy, T_prec, T_rec, T_F1, F_prec, F_rec, F_F1 = get_results()
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                saved_T_prec = T_prec
                saved_T_rec = T_rec
                saved_T_F1 = T_F1
                saved_F_prec = F_prec
                saved_F_rec = F_rec
                saved_F_F1 = F_F1
                if os.path.exists("../fc_model"):
                    delete_dir("../fc_model")
                os.rename("../output", "../fc_model")
                os.makedirs("../output")
            else:
                empty_dir("../output")
        delete_dir("../output")
        print('T_acc:', highest_accuracy, ' T_prec:', saved_T_prec, ' T_rec:', saved_T_rec, ' T_F1:', saved_T_F1)
        print('F_acc:', highest_accuracy, ' F_prec:', saved_F_prec, ' F_rec:', saved_F_rec, ' F_F1:', saved_F_F1)
        output_data = []
        output_data.append('T_acc:' + str(highest_accuracy) + ' T_prec:' + str(saved_T_prec) + ' T_rec:' + str(saved_T_rec) + ' T_F1:' + str(saved_T_F1))
        output_data.append('F_acc:' + str(highest_accuracy) + ' F_prec:' + str(saved_F_prec) + ' F_rec:' + str(saved_F_rec) + ' F_F1:' + str(saved_F_F1))
        write_file("../test_results.txt", output_data)
        
if __name__ == "__main__":

    flags.mark_flag_as_required("iteration")
    tf.app.run()
