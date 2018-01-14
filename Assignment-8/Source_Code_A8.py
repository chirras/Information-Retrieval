#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 19:35:37 2017

@author: satishreddychirra
"""


file_qrels = open('/Users/satishreddychirra/Document/Information Retrieval/IR A1/qrels.adhoc.51-100.AP89.txt', 'r')
qrels = {}
for line in file_qrels.readlines():
    l = line.split(' ')
    qrels[l[0],l[2]] = int(str(l[3]).replace('\n',''))



# BM25 
import re
import math
import elasticsearch.helpers
from operator import itemgetter
from collections import Counter
from nltk.stem import PorterStemmer
from elasticsearch import Elasticsearch

D = 84678
results = {}
qrels_m = {}
feature_list = []

es = Elasticsearch(timeout = 30)
query_file = open('/Users/satishreddychirra/Document/query')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    qry_score = Counter()
    for word in qry['Query Terms']:
        stemmer = PorterStemmer()
        word_s = ("%s" % (stemmer.stem(word)))
        res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match": {"text": word}},
                              "script_fields": {"index_df": {"script": {"lang": "groovy","inline": "_index['text']['" + word_s + "'].df()"}},
                              "index_tf": {"script": {"lang": "groovy","inline": "_index['text']['" + word_s + "'].tf()"}}}},
                              index="ap_dataset", doc_type="document", scroll=u"5m")
        
        res_list = []
        for i in res:
            res_list.append(i)

        
        tf = []
        doc_no = []
        doc_len = []
        bm25_score = []
        doc_freq = []
        for i in range(len(res_list)):
            tf.append(res_list[i]['fields']['index_tf'][0])
            doc_no.append(res_list[i]['_id'])
            doc_len.append(len(re.findall(r'\w+', res_list[i]['_source']['text'])))
            doc_freq.append(res_list[i]['fields']['index_df'][0])
            bm25_score.append((math.log((D+0.5)/(doc_freq[i]+0.5))) * ((tf[i] + 1.2*tf[i])/(tf[i] + 1.2*((1-0.75)+ 0.75 * (doc_len[i]/441)))))

        results.clear()
        results = dict(zip(doc_no, bm25_score))

        qry_score = Counter(results) + qry_score

    score = dict(qry_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)

    feat_list = []
    for i in range(len(score_list)):
        if(((qry["Query No"], score_list[i][0])) in qrels.keys()):
            feat_list.append((qry["Query No"], score_list[i][0], score_list[i][1]))
    
    qrels_m.clear()
    for k in score:
        qrels_m[qry["Query No"],k] = 0
    
    for k in qrels:
        if(k[0] == qry["Query No"] and k not in qrels_m):
            feat_list.append((qry["Query No"], k[1], qrels_m[min(qrels_m, key=qrels_m.get)]))
            
    for i in range(1000):  
        if(((qry["Query No"], score_list[i][0])) not in qrels.keys() and score_list[i][0] in score.keys() and score_list[i][0] in score.keys()):
            feat_list.append((qry["Query No"], score_list[i][0], score_list[i][1]))

    for element in feat_list:
        feature_list.append(element)

#------------------------------------------------------------------------------

# Extracting and cleaning the data

qry_no = ['85', '59', '56', '71', '64', '62', '93', '99', '58', '77', '54', '87', '94', '100', '89', '61', '95', '68', '57', '97', '98', '60', '80', '63', '91']

import re
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
count_vec = CountVectorizer(stop_words = 'english', max_df = 0.95, min_df = 2)
lda = LatentDirichletAllocation(n_topics=20, max_iter=20, learning_method='online')

for element in qry_no:
    temp = [q for q in feature_list if q[0] == element]
    text = []
    for i in temp:
        res = elasticsearch.helpers.scan(es, {"_source": True, "query": { "match": { "docno": i[1]}}},
                              index="ap_dataset", doc_type="document", scroll=u"5m")
    
        res_list = []
        for i in res:
            res_list.append(i)
                
        text.append(res_list[0]['_source']['text'])
            
    text_clean = []   
    for i in text:
        text_w = re.findall(r'[0-9a-z]+(?:\.?[0-9a-z]+)+', i.lower())
        text_p = ''
        for i in text_w:
            text_p += i + ' '
        text_clean.append(text_p)
    
    term_mat_p1 =  count_vec.fit_transform(text_clean)   
    lda.fit(term_mat_p1)
    feature_names = count_vec.get_feature_names()

    # Top 10 words related to each topic
    file = open('/Users/satishreddychirra/Document/topic_words_p1.txt','a')
    file.writelines('Top words of 20 topics for query: ' + str(element) + '\n')
    for num, topic in enumerate(lda.components_):
        topic_words = " ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])
        file.writelines('Topic ' + str(num) + ': \n' + topic_words + '\n')
    file.close()
    
    # Top three topics for each document:
    lda_ft_p1 = lda.fit_transform(term_mat_p1)
    file = open('/Users/satishreddychirra/Document/results_dw_p1.txt','a')
    file.writelines('For Query Number: ' + str(element) + '\n')
    for i in range(len(temp)): 
        file.writelines(str(temp[i][1]) + ': \n') 
        file.writelines(str(lda_ft_p1[i].argsort()[::-1][:3]) + '\n')
    file.close()
       
    # Top three topics for query: 
    lda_comp = lda.components_
    file = open('/Users/satishreddychirra/Document/results_qw_p1.txt','a')   
    temp = {}
    for i in range(len(lda_comp)):
        temp[i] = sum(lda_comp[i])
    sorted_temp = sorted(temp.items(), key=operator.itemgetter(1), reverse=True)
    file.writelines('Top 3 topics of query ' + str(element) + ': \n')   
    file.writelines(str([sorted_temp[i][0] for i in range(len(sorted_temp)) if i < 3]) + '\n')
    file.close()



#------------------------------------------------------------------------------

### Part-B

count_vec = CountVectorizer(stop_words = 'english', max_df = 0.95, min_df = 2)
lda = LatentDirichletAllocation(n_topics=200, max_iter=5, learning_method='online')

res = elasticsearch.helpers.scan(es, {"_source": True, "query": { "match_all": {}}},
                              index="ap_dataset", doc_type="document", scroll=u"5m")

res_list = []
for i in res:
    res_list.append(i)

text = []
for i in range(len(res_list)):
    text.append(res_list[i]['_source']['text'])

document_no = []
for i in range(len(res_list)):
    document_no.append(str(res_list[i]['_id']).strip(' '))

text_clean = []   
for i in text:
    text_w = re.findall(r'[0-9a-z]+(?:\.?[0-9a-z]+)+', i.lower())
    text_p = ''
    for i in text_w:
        text_p += i + ' '
    text_clean.append(text_p)


term_mat =  count_vec.fit_transform(text_clean)   
lda.fit(term_mat)

lda_ft = lda.transform(term_mat)

# Fitting KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=25, random_state=0).fit(lda_ft)

# Assigning cluster numbers to documents
pred = kmeans.predict(lda_ft)

cluster_map = {}
for i in range(len(document_no)):
    cluster_map[document_no[i]] = pred[i]

# Gettting relavent documents
rel_qrels = {}
for k in qrels:
    if(k[0] in qry_no and qrels[k] == 1):
        rel_qrels[k] = qrels[k]

doc_tup = []
for k in rel_qrels:
    doc_tup.append((k[1],k[0]))

# Getting the combinations
import itertools
comb = [x for x in itertools.combinations(doc_tup, 2)]

for i in range(len(comb)):
    comb[i] = comb[i][0] + comb[i][1]



same_qry_clu = 0
diff_qry_clu = 0
same_clu_diff_qry = 0
diff_clu_same_qry = 0        

for k in comb:
    if(k[1] == k[3] and cluster_map[k[0]] == cluster_map[k[2]]):
        same_qry_clu += 1
    if(k[1] == k[3] and cluster_map[k[0]] != cluster_map[k[2]]):
        diff_clu_same_qry += 1
    if(k[1] != k[3] and cluster_map[k[0]] == cluster_map[k[2]]):
        same_clu_diff_qry += 1
    if(k[1] != k[3] and cluster_map[k[0]] != cluster_map[k[2]]):
        diff_qry_clu += 1




acc_per = (same_qry_clu+diff_qry_clu)/(same_clu_diff_qry+diff_clu_same_qry+same_qry_clu+diff_qry_clu)






















