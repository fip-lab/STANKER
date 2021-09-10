"""Sentences Representation of BERT."""

import modeling
import tokenization
import tensorflow as tf
import numpy as np


class BertRepresent(object):
    """Sentences Representation of BERT"""

    def __init__(self,
                 sess,
                 config_file,
                 vocab_file,
                 max_seq_length,
                 init_checkpoint,
                 do_lower_case=True,
                 is_training=False):
        """Constructor for BertRepresent.
        
        Args:
          sess: the currenct tensorflow session.
          config_file: the json file of configration about BERT model.
          vocab_file: the txt file of words dictionary for BERT.
          max_seq_length: a scalar,the maximum length you want to get.
          init_checkpoint: the BERT checkpoint file.
          do_lower_case: a bool,whether lower the English words,default is True.
          is_training: a bool,whether training BERT model,default is False.
        """
    
        bert_config = modeling.BertConfig.from_json_file(config_file)
        
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case)
        
        self.max_seq_length = max_seq_length
        
        self.input_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length])
        self.input_mask_origin = tf.placeholder(tf.int32, shape=[None, max_seq_length])
        self.input_mask_pruned = tf.placeholder(tf.int32, shape=[None, max_seq_length*max_seq_length])
        self.token_type_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length])
        self.bert_model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=self.input_ids,
            input_mask_origin=self.input_mask_origin,
            input_mask_pruned=self.input_mask_pruned,
            token_type_ids=self.token_type_ids,
            use_one_hot_embeddings=False)
        
        # initialize BERT model from checkpoint
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        
        self.sess = sess
    
    def get_sentences_representation(self, sentences):
        """Gets sentences representation.
        
        Args:
          sentences: a list shape as [batch_size],contains the strings
          need to transform.
          
        Returens:
          arraylist of shape [batch_size, embedding_size].
        """
      
        input_ids, input_mask_origin, input_mask_pruned, segment_ids = self.convert_to_ids(self.tokenizer, sentences, self.max_seq_length)
        
        representation_output = self.bert_model.get_pooled_output()
        
        representation_output_array = self.sess.run(representation_output, feed_dict={self.input_ids: input_ids, self.input_mask_origin: input_mask_origin, self.input_mask_pruned: input_mask_pruned, self.token_type_ids: segment_ids})
        return representation_output_array
        
    def get_gap_index(self, input_ids):
        gap_index = []
        for index in range(len(input_ids)):
            if input_ids[index] == 101 or input_ids[index] == 102:
                gap_index.append(index)
        return gap_index
        
    def get_interval_index(self, gap_index, index):
        for th in range(len(gap_index)):
            if th == len(gap_index) - 1:
                return None
            elif gap_index[th] < index <= gap_index[th + 1]:
                return [gap_index[th], gap_index[th + 1]]
            else:
               pass

    def convert_to_ids(self, tokenizer, sentences, max_seq_length):
        """Converts string to ids.
        
        Args:
          tokenizer: the FullTokenizer object.
          sentences: a list shape as [batch_size],contains the strings.
          max_seq_length: a scalar,the maximum length to keep.
          
        Returns:
          arraylist of shape [batch_size, max_seq_length]
        """
        
        output_ids = []
        output_mask_origin = []
        output_mask_pruned = []
        segment_ids = []
        for sentence in sentences:
            origin_tokens = []
            origin_tokens.extend(tokenizer.tokenize(sentence))
            if len(origin_tokens) > max_seq_length -2:
                origin_tokens = origin_tokens[0: max_seq_length -2]
            
            tokens = []
            segment = []
            segment_id = 0
            tokens.append("[CLS]")
            segment.append(segment_id)
            for token in origin_tokens:
                if token == "sep":
                    tokens.append("[SEP]")
                    segment.append(segment_id)
                    segment_id = (segment_id + 1) % 2
                else:
                    tokens.append(token)
                    segment.append(segment_id)
            tokens.append("[SEP]")
            segment.append(segment_id)
            cls_mask = [1] * len(tokens)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            while len(ids) < max_seq_length:
                ids.append(0)
                cls_mask.append(0)
                segment.append(0)
            
            mask_pruned = []
            mask_pruned.extend(cls_mask)
            
            gap_index = self.get_gap_index(ids)
            for index in range(1, len(ids)):
                interval_index = self.get_interval_index(gap_index, index)
                if interval_index is None:
                    mask_pruned.extend([0] * len(ids))
                else:
                    temp = []
                    for th in range(len(ids)):
                        if interval_index[0] < th <= interval_index[1]:
                            temp.append(1)
                        else:
                            temp.append(0)
                    mask_pruned.extend(temp)
            
            assert len(ids) == max_seq_length
            assert len(mask_pruned) == max_seq_length*max_seq_length
            assert len(segment) == max_seq_length
            
            output_ids.append(ids)
            output_mask_pruned.append(mask_pruned)
            output_mask_origin.append(cls_mask)
            segment_ids.append(segment)
        
        return output_ids, output_mask_origin, output_mask_pruned, segment_ids


def cos_sim(vector_a, vector_b):

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim
