# encoding=utf-8
import os
import tensorflow as tf

running_parameters_s = "--task_name=bincls "\
                       "--do_train=true "\
                       "--do_eval=false "\
                       "--data_dir=../data/s "\
                       "--vocab_file=../bert/vocab.txt "\
                       "--bert_config_file=../bert/bert_config.json "\
                       "--init_checkpoint=../bert/bert_model.ckpt "\
                       "--max_seq_length=512 "\
                       "--train_batch_size=6 "\
                       "--learning_rate=2e-5 "\
                       "--num_train_epochs=8.0 "\
                       "--output_dir=../lgam-bert_s"
                  
running_parameters_c = "--task_name=bincls "\
                       "--do_train=true "\
                       "--do_eval=false "\
                       "--data_dir=../data/c "\
                       "--vocab_file=../bert/vocab.txt "\
                       "--bert_config_file=../bert/bert_config.json "\
                       "--init_checkpoint=../bert/bert_model.ckpt "\
                       "--max_seq_length=512 "\
                       "--train_batch_size=6 "\
                       "--learning_rate=2e-5 "\
                       "--num_train_epochs=8.0 "\
                       "--output_dir=../lgam-bert_c"
       
       
def main(_):
    os.system("python3 run_classifier.py " + running_parameters_s) 
    os.system("python3 present_data.py --data_file=../data/s/train.csv --output_file=../data/s/train_embedding.csv --model_path=../lgam-bert_s")
    os.system("python3 present_data.py --data_file=../data/s/test.csv --output_file=../data/s/test_embedding.csv --model_path=../lgam-bert_s")    
    os.system("python3 run_classifier.py " + running_parameters_c)
    os.system("python3 present_data.py --data_file=../data/c/train.csv --output_file=../data/c/train_embedding.csv --model_path=../lgam-bert_c")
    os.system("python3 present_data.py --data_file=../data/c/test.csv --output_file=../data/c/test_embedding.csv --model_path=../lgam-bert_c")
    os.system("python3 concatenate.py --data_a_file=../data/s/train_embedding.csv --data_b_file=../data/c/train_embedding.csv --output_file=../data/train.csv")
    os.system("python3 concatenate.py --data_a_file=../data/s/test_embedding.csv --data_b_file=../data/c/test_embedding.csv --output_file=../data/test.csv")
    os.system("python3 train_fc.py --iteration=100")
                     
                     
if __name__ == "__main__":
    tf.app.run()