#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import re
import os
import math
import pickle
import copy
import bz2
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 


# In[2]:


files = []
words_list1 = []
directory = r'20_newsgroups\\'
for entry in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, entry)):
        files.append(entry)

docs = []
directory = r'20_newsgroups\\'

mstr_dict_dict = {}
all_doc_ids = []
list_vocab_words = []
dict_folder_docs = {}

for fol in files:
    temp_dir = os.path.join(directory, fol)
    docs = []
    complete_doc_loc = []
    for entry in os.listdir(temp_dir):
        if os.path.isfile(os.path.join(temp_dir, entry)):
            docs.append(entry)
            doc_loc = os.path.join(fol, entry)
            complete_doc_loc.append(doc_loc)

    all_doc_ids.extend(complete_doc_loc)
    dict_folder_docs[fol] = complete_doc_loc
    
    for dd in range(len(docs)):
        doc = docs[dd]
        full_path = os.path.join(temp_dir, doc)
        fp = open(full_path, "r")
        text = fp.read()
        fp.close()
        ll = text.split("\n\n")
        del ll[0]
        text = "\n\n".join(ll)
        text = text.lower()
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'[a-zA-Z]+[0-9]+', '', text)
        text = re.sub(r'[0-9]+[a-zA-Z]+', '', text)
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
        text = text.translate(translator)
#         text = re.sub(r'\d+', '', text)
        word_tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        text = [word for word in word_tokens if word not in stop_words]
#         lemmas = [ps.stem(word) for word in text]
        lemmas = [lemmatizer.lemmatize(word) for word in text]
        dict_doc = Counter(lemmas)
        temp_set = set(lemmas)
        unq_wrds = list(temp_set)
        list_vocab_words.extend(unq_wrds)
#         doc_location = os.path.join(fol, doc)
        doc_location = complete_doc_loc[dd]
        mstr_dict_dict[doc_location] = copy.deepcopy(dict_doc)


# In[3]:


temp_set = set(list_vocab_words)
list_vocabulary = list(temp_set)
dict_nt = {}
for a in list_vocabulary:
    dict_nt[a] = 0
    for b in mstr_dict_dict:
        curr_dict = mstr_dict_dict[b]
        if curr_dict[a] != 0:
            dict_nt[a] = dict_nt[a] + 1

list_nt_values = []
for ab in dict_nt:
    list_nt_values.append(dict_nt[ab])

max_nt = max(list_nt_values)


# In[4]:


count_docs = len(all_doc_ids)
count_vocab = len(list_vocabulary)
# data_matrix_vectors = [[0 for i in range(count_vocab)] for j in range(count_docs)]
dict_matrix_vectors = {}


# In[5]:


for c in range(count_docs):
    curr_doc = all_doc_ids[c]
    curr_doc_dict = mstr_dict_dict[curr_doc]
    dict_matrix_vectors[curr_doc] = []
    for d in range(count_vocab):
        curr_wd = list_vocabulary[d]
        val_tf_log = math.log10(1 + curr_doc_dict[curr_wd])
        val_idf_max = math.log10(count_docs / (1 + dict_nt[curr_wd]))
        prod = val_tf_log * val_idf_max
        dict_matrix_vectors[curr_doc].append(prod)
#         data_matrix_vectors[c][d] = prod


# In[6]:


Picklefile3 = open('data_vocabulary', 'wb')
pickle.dump(list_vocabulary, Picklefile3)
Picklefile3.close()


# In[7]:


Picklefile1 = bz2.open('data_vectors', 'wb')
pickle.dump(dict_matrix_vectors, Picklefile1)
Picklefile1.close()


# In[8]:


Picklefile2 = open('dictionary_nt', 'wb')
pickle.dump(dict_nt, Picklefile2)
Picklefile2.close()


# In[ ]:




