import numpy as np 
import pandas as pd
import random
import collections

from gensim import corpora
from nltk.stem import WordNetLemmatizer

import gensim


#load data from file
train_df = pd.read_csv('all/train.tsv', sep = '\t', header = 0)
test_df = pd.read_csv('all/test.tsv', sep = '\t', header = 0)
raw_docs_train = train_df['Phrase'].values
raw_docs_test = test_df['Phrase'].values
sentiment_train = train_df['Sentiment'].values
num_labels = len(np.unique(sentiment_train))



print ('pre-processing train docs.........')
processed_docs_train = []
for doc in raw_docs_train:
  doc = doc.lower()
  doc = doc.replace('-', ' ')
  doc = doc.replace('\/', ' ')
  doc = doc.replace('/', ' ')
  doc = doc.replace('*', ' ')
  doc = doc.split()
  lemm = WordNetLemmatizer()
  lemmed = [lemm.lemmatize(w) for w in doc]
  processed_docs_train.append(lemmed)



print ('pre-processing test docs.........')
processed_docs_test = []
for doc in raw_docs_test:
  doc = doc.lower()
  doc = doc.replace('-', ' ')
  doc = doc.replace('\/', ' ')
  doc = doc.replace('/', ' ')
  doc = doc.replace('*', ' ')
  doc = doc.split()

  lemm = WordNetLemmatizer()
  lemmed = [lemm.lemmatize(w) for w in doc]
  processed_docs_test.append(lemmed)


processed_docs_all = processed_docs_train+processed_docs_test
processed_docs_all.append(["PAD"])
dictionary = corpora.Dictionary(processed_docs_all)
dictionary_size = len(dictionary.keys())
print( "dictionary size: ", dictionary_size)

print ("converting to token ids......")
word_id_train, word_id_len = [], []
for doc in processed_docs_train:
  word_ids = [dictionary.token2id[word] for word in doc]
  word_id_train.append(word_ids)
  word_id_len.append(len(word_ids))

word_id_test = []
for doc in processed_docs_test:
  word_ids = [dictionary.token2id[word] for word in doc]
  word_id_test.append(word_ids)
  word_id_len.append(len(word_ids))

seq_len = np.round((np.mean(word_id_len)+2*np.std(word_id_len))).astype(int)
print ("the average sentence length is: ", seq_len)

#pad or truncate the sentences
pad_id = dictionary.token2id['PAD']
sequence_len_train, sequence_len_test = [], []
for i, sent in enumerate(word_id_train):
  if (len(sent) > seq_len):
    word_id_train[i] = sent[:seq_len]
    sequence_len_train.append(seq_len)
  else:
    sequence_len_train.append(len(sent))
    [sent.append(pad_id) for i in range(seq_len-len(sent))]

for i, sent in enumerate(word_id_test):
  if (len(sent) > seq_len):
    word_id_test[i] = sent[:seq_len]
    sequence_len_test.append(seq_len)
  else:
    sequence_len_test.append(len(sent))
    [sent.append(pad_id) for i in range(seq_len-len(sent))]


def load_glove():
  glove_dict = {}

  print ("loading GLove model...")
  f = open("../glove.6B/glove.6B.100d.txt", 'r', encoding="utf8")
  for line in f:
    splitline = line.split()
    word = splitline[0]
    word_vector =np.array([float(val) for val in splitline[1:]])
    glove_dict[word] = word_vector
  return glove_dict

glove_dict = load_glove()
word_embeddings = np.zeros((dictionary_size, 100))
# for key in glove_dict.keys():
#   print (key)
for word, id in dictionary.token2id.items():
  word_vector = glove_dict.get(word)
  if word_vector is not None:
    word_embeddings[id] = word_vector
  else:
    word_embeddings[id] = np.random.normal(0, 1, 100)


#convert the sentiment lables to ont-hot vectors
sentiment_vectors_train = []
for label in sentiment_train:
  onehot_label = [0,0,0,0,0]
  onehot_label[label] = 1
  sentiment_vectors_train.append(onehot_label)

print ("saving word embeddings and the text ids...")
np.save("word_embeddings.npy", word_embeddings)
np.save("word_id_train.npy", word_id_train)
np.save("word_id_test.npy", word_id_test)
np.save("sequence_len_train.npy", sequence_len_train)
np.save("sequence_len_test.npy", sequence_len_test)
np.save("sentiment_vectors_train.npy", sentiment_vectors_train)

