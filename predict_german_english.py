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


word_counter_save_path_english = '.word_counter_english.pkl'
word_counter_save_path_german = '.word_counter_german.pkl'

word_count_english = pkl_file_loader(word_counter_save_path_english)
word_count_german = pkl_file_loader(word_counter_save_path_german)

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
#model = define_model(vocab_size_german, vocab_size_english, max_size_german, max_size_english, 256)

#model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
#print(model.summary())

print("Loading Model: ")
model_path = "./German_English_model"
model_predict = tf.keras.models.load_model(model_path)

# Check its architecture
model_predict.summary()
#german_eval_sent = "Hallo"
german_eval_sent = "Fire"

# Preprocessing Input eval #

encoded_germa_eval = []
# Tokenising
print("Mapping German Sentence to Integers")
if len(german_eval_sent.split()) > 1:
    for word in german_eval_sent.split():
        word = word.lower()
        if word in german_tokens:
            en_word = word_idx_german[word]
        else:
            en_word = 1
        encoded_germa_eval.append(en_word)

if len(german_eval_sent.split()) == 1:
    german_eval_sent = german_eval_sent.lower()
    if german_eval_sent in german_tokens:
        en_word = word_idx_german[german_eval_sent]
    else:
        en_word = 1
    encoded_germa_eval.append(en_word)

#print(word_idx_german["hallo"])
# print((idx_word_german[1]))
# print(idx_word_german[361])
# print(idx_word_german[16593])
# print(idx_word_german[19672])
encoded_germa_eval =np.array(encoded_germa_eval)
encoded_germa_eval = np.expand_dims(encoded_germa_eval,-1)
print("Encoded_german_Eval: ")
print(encoded_germa_eval)
print("Encoded_german_Eval Shape: ")
print(encoded_germa_eval.shape)
# Padding
pad_encoded_german = pad_sequences(encoded_germa_eval , maxlen=max_size_german)
print("Padded Encoded German Eval: ")
print(pad_encoded_german)
print("Padded Encoded German Eval Shape: ")
print(pad_encoded_german.shape)
translation = model_predict.predict_class(pad_encoded_german)
print("Translation")
print(translation)
print(np.array(translation).shape)
integers = np.argmax(translation , axis = -1)
print("-------------------------------")
print(integers)
print(np.array(integers).shape)
tr = []
for batch in integers:          #1st dim
    for int_1 in batch:        #2nd dim
        eng_trans = idx_word_english[int_1]
        tr.append(eng_trans)
    print(tr)

print("----------------------------------------------------------")
#print(idx_word_english[13773])
#print(idx_word_english[1])
# print(idx_word_english[6868])
# print(idx_word_english[17707])
# print(idx_word_english[61551])
# print(idx_word_english[9173])

print(word_count_english["to"])
print(max(word_count_english.values()))