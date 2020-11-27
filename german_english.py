
import numpy as np
import tensorflow as tf
import sys
import os
import string
import re
import pickle
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,LSTM,RepeatVector,TimeDistributed,Embedding
from tensorflow.keras.models import Sequential
import sparse

from keras.utils import to_categorical
#
# localtime = time.asctime( time.localtime(time.time()) )
# time = localtime.split()
# month = time[1]
# date = time[2]
# year = time[-1]
# tm = time[3]
# date_form = month+"_"+date+"_"+year
# print("time: ",date_form)
#
# old_stdout = sys.stdout
#
# log_file = open(date_form + "_" + "message_1.log", "a")
#
# sys.stdout = log_file
#
# print("Log File Starts :")
# print("time: ",date_form + "_" + tm)

# Loading the File
#
# def file_loader(file_path):
#     """
#     Loads the file
#     :param file_path: path to file
#     :return: file content
#     """
#     print("loading file")
#     with open(file_path,'r', encoding='utf-8') as f: # encoding is used to get string object and not bytes object
#         content = f.read()
#     print("load successful")
#     return content
#
# filepath = "./dataset/deu.txt"
# data = file_loader(filepath)
#
# print(type(data))
# #print(len(data))
#
# def get_pairs(data_file):
#     """
#     Preprocess the file and load the pairs
#     :param data_file: Content of the file
#     :return: pair(english,german)
#     """
#     pair_list = []
#     print("get data-tuple")
#     lines = data_file.strip().split("\n")
#     for line in lines:
#         pair = line.split("\t")
#         pair_list.append(pair)
#     return np.array(pair_list)
#
# data_pair = get_pairs(data)
# print(data_pair[0])
# print(data_pair.shape)
# print(data_pair[:10])
# print(len(data_pair))
#
# ### Preprocessing Pairs ##############
#
# def clean_pairs(lines):
#     """
#
#     :param lines: data_pairs (list)
#     :return: cleaned_pairs (list)
#     """
#     cleaned = list()
#     # prepare regex for char filtering
#     re_print = re.compile('[^%s]' % re.escape(string.printable))
#     # prepare translation table for removing punctuation
#     table = str.maketrans('', '', string.punctuation)
#     for pair in lines:
#         clean_pair = list()
#         for word in pair:
#             # Lowercasing_each word
#             word = word.lower()
#             # remove punctuation from word
#             table = str.maketrans("","",string.punctuation)
#             word = word.translate(table)
#             # remove non-printable chars form each token
#             word = re_print.sub('', word)
#             if word.isalpha():
#                 word = word
#
#             clean_pair.append(word)
#         cleaned.append((clean_pair))
#
#     return np.array(cleaned)
#
# clean_data = clean_pairs(data_pair)
# print(clean_data.shape)
# print(clean_data[:10])
#



def pkl_file_saver(filepath_save,content_save):
    """
    Saves file Pickle (only)
    :param content_save: What to Save
    :param filepath_save: Where to Save
    :return:
    """
    print("Saving File..")
    with open(filepath_save,'wb') as f:
        pickle.dump(content_save,f)
    print(" Saved Successfully....")
# file_path_to_save = "./dataset/cleaned_deu.pkl"
# pkl_file_saver(file_path_to_save,clean_data)
#


def pkl_file_loader(filepath):
    """

    :param filepath: Extracting file path
    :return: file
    """
    print("Loading Pickle File")

    with open(filepath,'rb') as f:
        file = pickle.load(f)
    print("File Loaded Successfully")

    return file

file_path_to_load = "./dataset/cleaned_deu.pkl"
clean_data = pkl_file_loader(file_path_to_load)

clean_data = np.array(clean_data)

# print(clean_data[:10])
# print(clean_data.shape)
#
# # data = []
# # for element in clean_data:
# #     sent_data = []
# #     english,german = np.array_split(element,2)
# #     # Not what I want .. I want pairs to be preserved instead use of identifier as in the case below
# #     # data.append("English: %s"%(" ".join([word for word in english])))
# #     # data.append("German: %s"%(" ".join([word for word in german])))
# #     english_sent = " ".join([word for word in english])
# #     german_sent = " ".join([word for word in german])
# #     sent = english_sent + ", " + german_sent
# #     sent_data.append(sent)
# #
# # print(data[:10])
# # print(len(data)//2)
#
#
#
#
#
#
#
# ## Creating a dataset obj ###
#
def dataset_builder(clean_data):
    """
    Builds a dataset object of clean data
    :param clean_data: processed clean data
    :return: dataset_obj
    """
    print("Building Dataset object")
    dataset = tf.data.Dataset.from_tensor_slices(clean_data)
    return dataset

dataset = dataset_builder(clean_data)
print(type(dataset))
dataset.batch(64 , drop_remainder=True)
print(dataset.element_spec)
print(list(dataset.take(1).as_numpy_iterator()))
#
# ## Convertin byte object to String obj ##
#
# # for ls in dataset.as_numpy_iterator():
# #     words_ls = ls.split()
# #     words = [str(word) for word in words_ls]
# #
# #
# # print(list(dataset.take(1).as_numpy_iterator()))
#
#
# # If tokenise over word it tokenises at char_level because it splits the input in this case word --> so translates to char
# # One Solution input lines insted of list
#

#
# word_count_english = {}
# word_count_german = {}
# translation_counter = 0
#
# for element in dataset.as_numpy_iterator():  # Batch file # 64 Translations
#     english_trans,german_trans = np.array_split(element,2)    # file # 1 Translation
#     print("Translation Acccessed")
#     translation_counter += 1
#     print(translation_counter)
#     # print(element)
#     # print(english_trans)
#     # print(german_trans)
#     print("--------------------")
#     for english_sent in english_trans:
#         english_sent = english_sent.decode('UTF-8')
#         for word in english_sent.split():
#             if word not in word_count_english:
#                         word_count_english[word] = 0
#             word_count_english[word] += 1
#
#     for german_sent in german_trans:
#         german_sent = german_sent.decode('UTF-8')      # Decoding german_trans (byte-like) to german_sent(string like)
#         for word in german_sent.split():
#             if word not in word_count_german:
#                 word_count_german[word] = 0
#             word_count_german[word] += 1
#
#
#
#
#
#

word_counter_save_path_english = '.word_counter_english.pkl'
word_counter_save_path_german = '.word_counter_german.pkl'
#
# pkl_file_saver(word_counter_save_path_english,word_count_english)
# pkl_file_saver(word_counter_save_path_german,word_count_german)

word_count_english = pkl_file_loader(word_counter_save_path_english)
word_count_german = pkl_file_loader(word_counter_save_path_german)
#
# word_count_english = {k: v for k, v in sorted(word_count_english.items(), key=lambda item: item[1], reverse=True)}
# word_count_german = {k: v for k, v in sorted(word_count_german.items(), key=lambda item: item[1], reverse=True)}

# print(word_count_english)
# print(word_count_german)
print("English Tokens: ")
print(len(word_count_english.keys()))
print("German Tokens: ")
print(len(word_count_german.keys()))

english_tokens = sorted(list(word_count_english.keys()))
german_tokens = sorted(list(word_count_german.keys()))


def create_mappings(tokens, debuggining_info=True):
    """

    :param tokens: sorted tokens list
    :return: word_to_idx and idx_to_word
    """
    word_idx = {}
    for i, word in enumerate(tokens, start=2):  # Leave 0 for padding and 1 for unkown tokens
        word_idx[word] = i
    word_idx["<UNK>"] = 1

    idx_word = {}
    for i, word in enumerate(tokens, start=2):
        idx_word[i] = word
    idx_word[1] = "<UNK>"

    if debuggining_info:
        print("Length of Word index: ", len(word_idx.items()))  # num_words + 1 (Unknown Token / 0 padding)
        print("Length of index word mapping: ", len(idx_word.items()))  # num_words + 1 (UNK TOken/ (0--padding)
        print("Length of tokens: ", len(tokens))  # num_words
        #print("length of updated word counter : ", len(updated_word_count.items()))  # num_words

    return word_idx, idx_word

word_idx_english , idx_word_english = create_mappings(english_tokens)
word_idx_german , idx_word_german = create_mappings(german_tokens)

vocab_size_english = len(english_tokens) + 2  # (num_words + 1 (unknown token) + 1 (0 padding to make all equal length)

vocab_size_german = len(german_tokens) + 2  # (num_words + 1 (unknown token) + 1 (0 padding to make all equal length)
print("Vocab size English : %s"%vocab_size_english)
print("Vocab size German : %s" %(vocab_size_german))


# On run on 25_nov file was appending at word level and not at translation level this is now updated..on 26_NOv
#
# counter = 0
# english_counter = 0
# german_counter = 0
# tokenised_english_content = []
# tokenised_german_content = []
#
# for element in dataset.as_numpy_iterator():  # Batch file # 64 Translations
#     english_trans,german_trans = np.array_split(element,2)    # file # 1 Translation
#     print("Translation Acccessed")
#     counter += 1
#     print(counter)
#     # print(element)
#     # print(english_trans)
#     # print(german_trans)
#     print("--------------------")
#     for english_sent in english_trans:
#         english_sent = english_sent.decode('UTF-8')         # Decoding german_trans (byte-like) to german_sent(string like)
#         encoded_english_translation_content = []
#         english_counter += 1
#         for word in english_sent.split():
#                 if word in english_tokens:
#                     encoded_english_translation_content.append(word_idx_english[word])
#                 else:
#                     encoded_english_translation_content.append(1)
#         tokenised_english_content.append(encoded_english_translation_content)
#         print("Actual Content Englsih: ")
#         print(english_sent)
#         print("Encoded Content English: ")
#         print(encoded_english_translation_content)
#         print("--------------------------")
#
#     for german_sent in german_trans:
#         german_sent = german_sent.decode('UTF-8')  # Decoding german_trans (byte-like) to german_sent(string like)
#         encoded_translation_german_content = []
#         german_counter += 1
#         for word in german_sent.split():
#             if word in german_tokens:
#                 encoded_translation_german_content.append(word_idx_german[word])
#             else:
#                 encoded_translation_german_content.append(1)
#         tokenised_german_content.append(encoded_translation_german_content)
#
#         print("Actual Content German: ")
#         print(german_sent)
#         print("Encoded Content: ")
#         print(encoded_translation_german_content)
#
# print(counter)
# print(english_counter)
# print(german_counter)
#

file_path_tokens_content_english = ".tokenised_content_english.pkl"
file_path_tokens_content_german = ".tokenised_content_german.pkl"
#
# pkl_file_saver(file_path_tokens_content_english,tokenised_english_content)
# pkl_file_saver(file_path_tokens_content_german,tokenised_german_content)


tokens_content_english = pkl_file_loader(file_path_tokens_content_english)
tokens_content_german = pkl_file_loader(file_path_tokens_content_german)

for element in dataset.take(10).as_numpy_iterator():
    print(element)
print(tokens_content_english[:10])
print(tokens_content_german[:10])
print(len(tokens_content_english))      # Content Length Cna be different as no translation is word to word , it also accounts for difference in grammatical structures of two languages
print(len(tokens_content_german))
# print(np.array(tokens_content_english).shape)
# print(np.array(tokens_content_german).shape)

tmp_counter = 0
max_size_english = 0
for element in tokens_content_english:
    tmp_counter += 1
    if max_size_english < len(element):
        max_size_english = len(element)

tmp_counter = 0
max_size_german = 0
for element in tokens_content_german:
    tmp_counter += 1
    if max_size_german < len(element):
        max_size_german = len(element)

print(max_size_english)
print(max_size_german)

pad_english_sentence = pad_sequences(tokens_content_english,max_size_english)
pad_german_sentence = pad_sequences(tokens_content_german,max_size_german)
print("Padded EncodedEnglish Sentences: ")
print(pad_english_sentence[:10])
print("Padded Encoded German Sentences")
print(pad_german_sentence[:10])
print("Padded English Sentence Shape: ")
print(pad_english_sentence.shape)
print("Padded German Sentence Shape: ")
print(pad_german_sentence.shape)


## Converting Targets to one-hot representation as each word will be a prediction (different class)




print("Clean Data: ")
print(clean_data[:5])
print(clean_data.shape)

#
# #on_target = np.zeros((pad_english_sentence.shape[0],pad_english_sentence.shape[1],vocab_size_english) , dtype='np.uint8')
# sparse_arr = sparse.zeros((pad_english_sentence.shape[0],pad_english_sentence.shape[1],vocab_size_english))
# sparse_arr = sparse.COO.maybe_densify(sparse_arr)
#
# cnt_sequence = 0
# for sequence in pad_english_sentence:
#     print("Sequence Accessed : ")
#     print(cnt_sequence)
#     print(sequence)
#     on_seq = []
#     num_cnt = 0
#     for num in sequence:
#         num_cnt += 1
#         if num != 0:
#             sparse_arr[:,:,num] = 1
#     print(num_cnt)
#
# pkl_file_saver(".one_hot_target_english.pkl",sparse_arr)
#
#
# # one hot encode target sequence
# def encode_output(sequences, vocab_size):
#     """
#
#     :param sequences: Target_padded Sequence
#     :param vocab_size: Target_vocab_size
#     :return: Padded_encoded_on_hot output
#     """
#     cnt = 0
#     ylist = list()
#     for sequence in sequences:
#         print("Sequence Accessed: ")
#         cnt += 1
#         print(sequence)
#         print("---------------------")
#         encoded = to_categorical(sequence, num_classes=vocab_size)
#         print(encoded)
#         ylist.append(encoded)
#     y = np.array(ylist)
#     y = np.reshape(y,(sequences.shape[0], sequences.shape[1], vocab_size))
#     return y

# oh_english = encode_output(pad_english_sentence,vocab_size_english)
# pkl_file_saver(".oh_english.pkl",oh_english)
#
# print(oh_english.shape)
from keras.layers import Lambda
# We will use `one_hot` as implemented by one of the backends
from keras import backend as K


def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                         num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))


#
# # define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    """
    Defines and Creates the model
    :param src_vocab: German_vocab_size
    :param tar_vocab: English_vocab_size
    :param src_timesteps: Max_len_German_input
    :param tar_timesteps: Max_len_English_output
    :param n_units: For Representation
    :return:
    """
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))  # Repeates vector target time_steps time (important to fuse Encoder and Decoder)
    model.add(LSTM(n_units, return_sequences=True))
    #model.add(OneHot(input_dim=tar_vocab,input_length=tar_timesteps))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model

# define model
model = define_model(vocab_size_german, vocab_size_english, max_size_german, max_size_english, 256)

model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())



# One Hot Representation of Target Vectors

import tensorflow as tf

english_ds = tf.data.Dataset.from_tensor_slices(pad_english_sentence)
english_ds = english_ds.batch(64)

def one_hot(ds):
    return to_categorical(ds, num_classes=vocab_size_english)

english_ds = english_ds.map(one_hot())

german_ds = tf.data.Dataset.from_tensor_slices(pad_german_sentence)
german_ds = german_ds.batch(64)

# batch_cnt = 0
# file_cnt = 0
# for english_batch in english_ds.as_numpy_iterator():  # 64 batch files
#     print("Batch Accessed: ")
#     batch_cnt += 1
#     print(batch_cnt)
#     for english_sent in english_batch:
#         print("File Accessed: ")
#         file_cnt += 1
#         print(file_cnt)
#         #print(english_sent.shape)
#         #print(english_sent)
#




#
# checkpoint_dir = "./German_English_Trans_checkpoints"
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
#
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_prefix,
#     save_weights_only=True)
#
# #history = model.fit(pad_german_sentence,pad_english_sentence,epochs=30,batch_size = 64,callbacks = [checkpoint_callback],validation_split = 0.2)


# Target Sequence is ENglish in our case
# one hot encode target sequence

# #
# sys.stdout = old_stdout
#
# log_file.close()


# print(idx_word_german[10])
# print(idx_word_german[11])
#
# pad_english_sentences = pad_sequences(tokens_content_english)
# pad_german_sentences = pad_sequences(tokens_content_german)
#
# print(pad_english_sentences.shape)
# print(pad_german_sentences.shape)





