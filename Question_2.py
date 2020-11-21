#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import re
import os
import math
import statistics
import pickle
import copy
import numpy
import matplotlib.pyplot as plt
import bz2
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.manifold import TSNE
lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 


# In[2]:


files = []
directory = r'20_newsgroups\\'
for entry in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, entry)):
        files.append(entry)

docs = []
directory = r'20_newsgroups\\'
all_doc_ids = []
dict_folder_docs = {}

for fol in files:
    temp_dir = os.path.join(directory, fol)
    complete_doc_loc = []
    for entry in os.listdir(temp_dir):
        if os.path.isfile(os.path.join(temp_dir, entry)):
            doc_loc = os.path.join(fol, entry)
            complete_doc_loc.append(doc_loc)

    all_doc_ids.extend(complete_doc_loc)
    dict_folder_docs[fol] = complete_doc_loc


# In[3]:


Picklefile1 = bz2.open('data_vectors', 'rb')
dict_matrix_vectors = pickle.load(Picklefile1)

Picklefile2 = open('dictionary_nt', 'rb')
dict_nt = pickle.load(Picklefile2)

# list_val_nt = [dict_nt[z] for z in dict_nt]
# max_nt = max(list_val_nt)


# In[4]:


Picklefile3 = open('data_vocabulary', 'rb')
list_vocab_words = pickle.load(Picklefile3)
count_vocab = len(list_vocab_words)
count_docs_all = len(all_doc_ids)


# In[7]:


# print(len(dict_matrix_vectors[0]))
print(count_vocab)
# print(max_nt)
# print(files)
# print(all_doc_ids)
# for k in all_doc_ids:
#     print(k)


# In[8]:


print("Enter the value of k")
value_k = int(input())
value_alpha = 1
value_beta = 0.75
value_gamma = 0.25


# In[17]:


def do_preproceesing(text):
    text = text.lower()
    text = re.sub(r'[a-zA-Z]+[0-9]+', '', text)
    text = re.sub(r'[0-9]+[a-zA-Z]+', '', text)
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    text = text.translate(translator)
    word_tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    text = [word for word in word_tokens if word not in stop_words]
    query_tokens = [ps.stem(word) for word in text]
    return query_tokens

def make_vector_log(tokens_query):
    q_vector = numpy.zeros(count_vocab)
    word_counts_query = Counter(tokens_query)
    for i in range(count_vocab):
        curr_wd = list_vocab_words[i]
        val_tf_log = math.log10(1 + word_counts_query[curr_wd])
        idf_val = math.log10(count_docs_all / (1 + dict_nt[curr_wd]))
        prod = val_tf_log * idf_val
        q_vector[i] = prod
    return q_vector

def find_similarity(query_vector):
    norm_q_vector = numpy.linalg.norm(query_vector)
    list_cosine_sim = []
    for dc in range(len(all_doc_ids)):
        curr_doc_name = all_doc_ids[dc]
        doc_vector = dict_matrix_vectors[curr_doc_name]
        doc_numpy_vector = numpy.array(doc_vector)
        norm_doc_vector = numpy.linalg.norm(doc_numpy_vector)
        cosine_sim = (numpy.dot(query_vector, doc_numpy_vector)) / (norm_q_vector * norm_doc_vector)
        list_cosine_sim.append([all_doc_ids[dc] ,cosine_sim])
    return list_cosine_sim

#Now running the multiple iterations till the user wants to quit

def plot_PR_curve(query_no):
    count_relevant = 0
    list_precision = []
    list_recall = []
    global list_relevant
    global list_non_relevant
    global list_map_value
    list_precision_values = []
    for f in range(value_k):
        curr_doc = list_cosine_scores[f][0]
        if dict_ground_docs.get(curr_doc) is not None:
            count_relevant += 1
        value_precision = count_relevant / (f+1)
        value_recall = count_relevant / total_relevant
        list_precision.append(value_precision)
        list_recall.append(value_recall)
        if dict_ground_docs.get(curr_doc) is not None:
            list_precision_values.append(value_precision)
        
    if len(list_precision_values) == 0:
        list_map_value[query_no].append(0)
    else:
        list_map_value[query_no].append(statistics.mean(list_precision_values))
    
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(8,8))

    plt.plot(list_recall, list_precision, )
    plt.xlabel('Recall', fontsize = 14)
    plt.ylabel('Precision', fontsize = 14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

def plot_TSNE(vectors_queries):
    np_TSN_query = numpy.array(vectors_queries)
    array_2D = TSNE(n_components=2).fit_transform(np_TSN_query)
    list_colours = ["red", "green", "blue", "yellow", "black"]
    list_markers = ["+", "*", "^", "X", "d"]
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(7, 7))
    for r in range(len(array_2D)):
        curr_cord = array_2D[r]
        string_iter = "iteration" + str(r)
        plt.scatter(curr_cord[0], curr_cord[1], label= string_iter, color= list_colours[r],  marker= list_markers[r], s=40) 
    
    plt.xlabel('X-axis', fontsize = 14)
    plt.ylabel('Y-axis', fontsize = 14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend() 
    plt.show()


# In[18]:


print("How many queries you want to enter:")
count_queries = int(input())
list_map_value = [[] for kh in range(count_queries)]


# In[19]:


for y in range(count_queries):
    print(f"Enter the initial Query{(y + 1)}")
    query = input()
    list_query = do_preproceesing(query)
    print(f"\nEnter the ground truth folder for Query{(y + 1)}:")
    for a in range(len(files)):
        print(f"Enter {(a+1)} for {files[a]}")

    ground_sno = int(input())
    dict_ground_docs = {}
    ground_folder = files[ground_sno - 1]
    list_TSN_queries = []
    for b in dict_folder_docs[ground_folder]:
        dict_ground_docs[b] = 1
    
    total_relevant = len(dict_ground_docs)
    count_iteration = 0
    dict_already_marked = {}
    #Now for each of the queries running the iterations till the user wants to stop the iterations
    vector_query = make_vector_log(list_query)
    flag_feedback = 1
    while True:
        list_TSN_queries.append(vector_query.tolist())
        list_cosine_scores = find_similarity(vector_query)
        list_cosine_scores.sort(key = lambda x :x[1], reverse = True)
        
        list_relevant = []
        list_unrelevant = []
        #Calling the function defined above to plot PR curve
        print("The PR curve is:")
        plot_PR_curve(y)
        if count_iteration != 0:
            print("Enter 1 to continue feedback iterations and 2 to stop feedback iterations")
            flag_feedback = int(input())
            
            if flag_feedback == 2:
                print(f"The top {value_k} documents after {count_iteration} iteration are:")
                for e in range(value_k):
                    print(f"{e+1}. {curr_docum}")
                break
                
        print("Enter the value of p in p% to be marked as relevant")
        p_percent = int(input())
        
        #Now doing the relevance feedback process of marking p% docs as relevant
        count_p_percent = int((p_percent / 100) * value_k)
        count_marked = 0
        dict_mark_this_iteration = {}
        list_relevant_in_k = []
        list_non_rel_in_k = []
        for f in range(value_k):
            curr_dc = list_cosine_scores[f][0]
            #Now checking if the document belongs to the ground truth folder and 
            #it has not been marked relevant in the previous iterations
            if dict_already_marked.get(curr_dc) is None:
                #If part for docs belong to ground truth docs
                if dict_ground_docs.get(curr_dc) is not None:
                    dict_already_marked[curr_dc] = 1
                    dict_mark_this_iteration[curr_dc] = 1
                    list_relevant_in_k.append(curr_dc)
                    count_marked += 1
                else:
                    list_non_rel_in_k.append(curr_dc)
                    
            if count_marked == count_p_percent:
                break
        
        #Now there may be case that top p% are found before scanning whole list,
        #So finding remaining non-relevant docs
        if f < value_k:
            start_val = f - 1
            for gj in range(start_val, value_k):
                curr_dc = list_cosine_scores[gj][0]
                if dict_already_marked.get(curr_dc) is None:
                    list_non_rel_in_k.append(curr_dc)
                    
        #Now printing the top k documents
        if count_iteration == 0:
            print(f"The top {value_k} documents for initial query are:")
        else:
            print(f"The top {value_k} documents after {count_iteration} iteration are:")
        for e in range(value_k):
            curr_docum = list_cosine_scores[e][0]
            if dict_mark_this_iteration.get(curr_docum) is not None:
                print(f"{e+1}. {curr_docum}*")
            else:
                print(f"{e+1}. {curr_docum}")
        
        count_relevant_in_k = len(list_relevant_in_k)
        count_non_relevant_in_k = len(list_non_rel_in_k)
        #Now finding the updated query
        term1 = vector_query * value_alpha
        #Finding the sum of relevant doc vectors
        summation_rel_vectors = numpy.zeros(count_vocab)
        for h in list_relevant_in_k:
            c_dc = dict_matrix_vectors[h]
            vector_rel_doc = numpy.array(c_dc)
            summation_rel_vectors = numpy.add(summation_rel_vectors, vector_rel_doc)
        prd_val = value_beta / count_relevant_in_k
        term2 = summation_rel_vectors * prd_val
        
        #Finding the sum of non-relevant doc vectors
        summation_non_rel_vectors = numpy.zeros(count_vocab)
        for l in list_non_rel_in_k:
            c_dc_n = dict_matrix_vectors[l]
            vector_NR_in_k = numpy.array(c_dc_n)
            summation_non_rel_vectors = numpy.add(summation_non_rel_vectors, vector_NR_in_k)
        
        prd_val_2 = value_gamma / count_non_relevant_in_k
        term3 = summation_non_rel_vectors * prd_val_2 
        
        updated_query_vector = numpy.add(term1, term2)
        updated_query_vector = numpy.subtract(updated_query_vector, term3)
        vector_query = updated_query_vector
        count_iteration += 1
    #Caaling the function defined above to plot TSNE curve
    print(f"\nThe TSNE curve for Query {(y+1)} is: ")
    plot_TSNE(list_TSN_queries)
        


# In[20]:


len_map_precision = [len(f) for f in list_map_value]
min_iterations = min(len_map_precision)
for r in range(min_iterations):
    list_mean_values = [li[r] for li in list_map_value]
    value_MAP = sum(list_mean_values) / count_queries
    if r == 0:
        print(f"The MAP value for the {count_queries} queries for initial query is iteration is {value_MAP}")
    else:
        print(f"The MAP value for the {count_queries} queries after {r+1} feedback iteration is {value_MAP}")


# In[ ]:





# In[ ]:




