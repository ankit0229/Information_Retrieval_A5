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
import random
import numpy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from tqdm import tqdm
lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 


# In[2]:


list_list_train_docs = []
list_list_test_docs = []
list_dict_feat_tf_idf = []
list_dict_feat_mi = []
list_dict_vectors_tfidf = []
list_dict_vectors_mi = []
list_list_features_vocab_tfidf = []
list_list_features_vocab_mi = []
list_dict_dict_tf_class = []


# ### Reading the documents of folder and performing pre-processing steps. Also preparing the required dictionaries

# In[3]:


files = []
words_list1 = []
directory = r'20_newsgroups\\'
for entry in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, entry)):
        files.append(entry)

docs = []
directory = r'20_newsgroups\\'
dict_folder_doc_mapping = {}
dict_doc_lemmas = {}
mstr_dict_dict = {}

for fol in files:
    temp_dir = os.path.join(directory, fol)
    docs = []
    complete_doc_loc = []
    dict_folder_doc_mapping[fol] = {}
    for entry in os.listdir(temp_dir):
        if os.path.isfile(os.path.join(temp_dir, entry)):
            docs.append(entry)
            doc_loc = os.path.join(fol, entry)
            complete_doc_loc.append(doc_loc)
            dict_folder_doc_mapping[doc_loc] = fol
            
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
        text = re.sub(r'[a-zA-Z]+[0-9]+', ' ', text)
        text = re.sub(r'[0-9]+[a-zA-Z]+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
        text = text.translate(translator)
        word_tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        text = [word for word in word_tokens if word not in stop_words]
#         lemmas = [ps.stem(word) for word in text]
        lemmas = [lemmatizer.lemmatize(word) for word in text]
        curr_doc = complete_doc_loc[dd]
        dict_doc_lemmas[curr_doc] = lemmas
        mstr_dict_dict[curr_doc] = Counter(lemmas)


# ### Defining function to split the documents into train and test set

# In[4]:


count_total_docs = len(mstr_dict_dict)
list_all_index = [ab for ab in range(count_total_docs)]
set_list_all_index = set(list_all_index)
list_all_doc_ids = list(mstr_dict_dict.keys())
all_doc_ids = list(mstr_dict_dict.keys())
copy_list_all_doc_ids = list_all_doc_ids
#Shuffling the list containg the doc ids
random.shuffle(copy_list_all_doc_ids)

def split_train_test(a, b):
    global count_train_docs
    count_train_docs = int((a / 100) * count_total_docs)
    global count_test_docs
    count_test_docs = count_total_docs - count_train_docs
    #Generating a list containing random indices of the docs to be used as training docs
    global list_train_index
    list_train_index = random.sample(list_all_index, count_train_docs)
    set_list_train_index = set(list_train_index)
    global list_test_index
    list_test_index = list(set_list_all_index.difference(set_list_train_index))
    #Taking the training docs from the list containing all doc ids
    global list_train_docs 
    list_train_docs = [copy_list_all_doc_ids[k] for k in list_train_index]
    global list_test_docs
    list_test_docs = [copy_list_all_doc_ids[j] for j in list_test_index]
    
    list_list_train_docs.append(list_train_docs)
    list_list_test_docs.append(list_test_docs)


# ### Defining the function to find the tf-idf values of terms in the training set and thus extracting features based on tf-idf values

# In[5]:


def feat_tf_idf():
    #Now finding the document frequency of each word of the complete vocabulary
    global list_dict_dict_tf_class
    dict_class_freq = {}
    dict_dict_tf_class = {}
    list_classes_tokens = []
    dict_tokens_classwise = {}
    global dict_dict_classwise
    dict_dict_classwise = {}
    for q in files:
        dict_tokens_classwise[q] = []
        dict_dict_classwise[q] = {}
        
    for p in list_train_docs:
        list_classes_tokens.extend(list(mstr_dict_dict[p].keys()))
        dict_tokens_classwise[dict_folder_doc_mapping[p]].extend(dict_doc_lemmas[p])
        dict_dict_classwise[dict_folder_doc_mapping[p]][p] = mstr_dict_dict[p]
    
    for q in dict_tokens_classwise:
        dict_dict_tf_class[q] = Counter(dict_tokens_classwise[q])
    
    list_dict_dict_tf_class.append(dict_dict_tf_class)
    
    global list_classes_vocab
    list_classes_vocab = list(set(list_classes_tokens))
    
    for a in list_classes_vocab:
        dict_class_freq[a] = 0
        for b in dict_dict_tf_class:
            if dict_dict_tf_class[b][a] != 0:
                dict_class_freq[a] = dict_class_freq[a] + 1
    
    #Now for each class storing the list of features accoding tothe tf-idf values
    global dict_feat_tf_idf_class
    dict_feat_tf_idf_class = {}
    count_classes = 5
    for c in dict_dict_tf_class:
        list_vocab_class = list(dict_dict_tf_class[c].keys())
        list_tf_idf_values = []
        for d in list_vocab_class:
            value_tf = math.log10(1 + dict_dict_tf_class[c][d])
            value_idf = math.log10(count_classes / (1 + dict_class_freq[d]))
            list_tf_idf_values.append(value_tf *  value_idf)
        
        #Sorting the features in descending order of tf-idf values
        list_pairs = sorted(zip(list_vocab_class, list_tf_idf_values), key = lambda x: x[1], reverse= True)
        list_sorted_features = [m for m,n in list_pairs]
        dict_feat_tf_idf_class[c] = list_sorted_features
    
    list_dict_feat_tf_idf.append(dict_feat_tf_idf_class)


# ### Defining the function to calculate the mutual independence value between each term of the vocab of training documents and each of the classes. 

# In[6]:


def feat_mi():
    global dict_feat_mi_class
    dict_feat_mi_class = {}
    for e in tqdm(files):
        list_mutual_values = []
        for f in tqdm(list_classes_vocab):
            #Now first finding the values for class = 1
            n_11 = 0
            n_01 = 0
            curr_dict = dict_dict_classwise[e]
            for g in curr_dict:
                if curr_dict[g][f] != 0:
                    n_11 += 1
                else:
                    n_01 += 1
            set_all_classes = set(files)
            set_curr_class = {e}
            set_other_classes = set_all_classes.difference(set_curr_class)
            #Now finding the values for class = 0
            n_10 = 0
            n_00 = 0
            for h in set_other_classes:
                curr_dict_2 = dict_dict_classwise[h]
                for i in curr_dict_2:
                    if curr_dict_2[i][f] != 0:
                        n_10 += 1
                    else:
                        n_00 += 1

            value_n = n_00 + n_01 + n_10 + n_11
            
            #Making the 2 * 2 matrix of mutual independence
            list_2d = [[n_11, n_10], [n_01, n_00]]
            np_list_2d = numpy.array(list_2d)
            #Finding the mutual indep value
            value_mutual_info = 0
            for i in range(2):
                for j in range(2):
                    t1 = np_list_2d[i][j] / value_n
                    if t1 == 0:
                        value_mutual_info += 0
                    else:
                        t2 = math.log2((value_n * np_list_2d[i][j]) / (numpy.sum(np_list_2d[i,:]) + numpy.sum(np_list_2d[j, :])))
                        value_mutual_info += (t1 * t2)

            list_mutual_values.append(value_mutual_info)
        #Sorting the features in the descending order of mutual independence values
        list_pairs_2 = sorted(zip(list_classes_vocab, list_mutual_values), key = lambda x: x[1], reverse= True)
        list_sorted_mi_features = [m for m,n in list_pairs_2]
        dict_feat_mi_class[e] = list_sorted_mi_features
    
    list_dict_feat_mi.append(dict_feat_mi_class)


# ### Defing the function to calculate the idf value of each term in training set

# In[7]:


def make_dict_nt(dict_feat, list_docs_training):
    global count_train_docs
    count_train_docs = len(list_docs_training)
    global dict_dict_tf_class
    dict_dict_tf_class = {}
    feat_list = []
    for gh in files:
        feat_list.extend(dict_feat[gh][:count_features+1])
    
    dict_dict_tf_class
    global list_features
    list_features = list(set(feat_list))
    
    global dict_nt
    dict_nt = {}
    for a in list_features:
        dict_nt[a] = 0
        for b in list_docs_training:
            curr_dict = mstr_dict_dict[b]
            if curr_dict[a] != 0:
                dict_nt[a] = dict_nt[a] + 1


# ### Defining the function to prepare the document vectors to be used in KNN algorithm

# In[8]:


def make_doc_vectors():
    global dict_matrix_vectors
    dict_matrix_vectors = {}
    count_docs = len(all_doc_ids)
    #Preparing the document vectors composed of tf-idf values of the terms
    for c in range(count_docs):
        curr_doc = all_doc_ids[c]
        curr_doc_dict = mstr_dict_dict[curr_doc]
        dict_matrix_vectors[curr_doc] = []
        for d in range(len(list_features)):
            curr_wd = list_features[d]
            val_tf_log = math.log10(1 + curr_doc_dict[curr_wd])
            val_idf = math.log10(count_train_docs / (1 + dict_nt[curr_wd]))
            prod = val_tf_log * val_idf
            dict_matrix_vectors[curr_doc].append(prod)


# ### Splitting the complete dataset using the 3 different split ratios. Also extracting the features using the 2 techniques seperately for the 3 different splits

# In[9]:


list_splits = [[50, 50],[70, 30],[80, 20]]
count_features = 100
for s in list_splits:
    split_train_test(s[0], s[1])
    feat_tf_idf()
    feat_mi()


# ### Calling the functions to make the document vectors according to the features extarcted for the differemt split ratios

# In[10]:


for t in (range(len(list_splits))):
    make_dict_nt(list_dict_feat_tf_idf[t], list_list_train_docs[t])
    list_list_features_vocab_tfidf.append(list_features)
    make_doc_vectors()
    list_dict_vectors_tfidf.append(dict_matrix_vectors)
    
    make_dict_nt(list_dict_feat_mi[t], list_list_train_docs[t])
    list_list_features_vocab_mi.append(list_features)
    make_doc_vectors()
    list_dict_vectors_mi.append(dict_matrix_vectors)


# ### Storing the pickles of required data

# In[11]:


Picklefile1 = open('list_dictionary_tf_idf_features', 'wb')
pickle.dump(list_dict_feat_tf_idf, Picklefile1)
Picklefile1.close()

Picklefile2 = open('files_list', 'wb')
pickle.dump(files, Picklefile2)
Picklefile2.close()

Picklefile3 = open('dictionary_doc_mapping', 'wb')
pickle.dump(dict_folder_doc_mapping, Picklefile3)
Picklefile3.close()

Picklefile4 = open('dictionary_master', 'wb')
pickle.dump(mstr_dict_dict, Picklefile4)
Picklefile4.close()

Picklefile5 = open('list_dictionary_tf_class', 'wb')
pickle.dump(list_dict_dict_tf_class, Picklefile5)
Picklefile5.close()

Picklefile6 = open('dictionary_doc_lemmas', 'wb')
pickle.dump(dict_doc_lemmas, Picklefile6)
Picklefile6.close()

Picklefile7 = open('list_dictionary_mi_features', 'wb')
pickle.dump(list_dict_feat_mi, Picklefile7)
Picklefile7.close()

Picklefile8 = bz2.open('list_dictionary_tfidf_vectors', 'wb')
pickle.dump(list_dict_vectors_tfidf, Picklefile8)
Picklefile8.close()

Picklefile9 = bz2.open('list_dictionary_mi_vectors', 'wb')
pickle.dump(list_dict_vectors_mi, Picklefile9)
Picklefile9.close()


# In[12]:


Picklefile10 = open('list_list_training_documents', 'wb')
pickle.dump(list_list_train_docs, Picklefile10)
Picklefile10.close()

Picklefile11 = open('list_list_testing_documents', 'wb')
pickle.dump(list_list_test_docs, Picklefile11)
Picklefile11.close()


# In[ ]:




