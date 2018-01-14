#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: satishreddychirra
"""


# Loading Out-Links
import elasticsearch.helpers
from operator import itemgetter
from elasticsearch import Elasticsearch
es = Elasticsearch(timeout = 30)
res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match_all": {}}}, index="team_asr", doc_type="document", scroll=u"5m", ignore=[400, 404])
res_list = []
for i in res:
    res_list.append(i)

# Out Links
crawled_outlinks = {}
for i in range(len(res_list)):
    crawled_outlinks[res_list[i]['_id']] = str(res_list[i]['_source']['out_links']).split('\n')

# Crawled Links  
crawled_links = []
for i in range(len(res_list)):
    crawled_links.append(res_list[i]['_id'])
    
# Document Length
text_w = ''
for i in range(len(res_list)):
    text_w += (res_list[i]['_source']['text'])

words_a = []
for word in text_w.split():
    words_a.append(word)

avg_doc_len = len(words_a)/len(crawled_links)


# Loading In-Links
import numpy as np
link_inlinks = np.load('url_inlinks.txt.npy').item()


# Pagerank
all_pages = len(crawled_outlinks)
S = [link for link in crawled_outlinks if crawled_outlinks[link][0] == '']
sink_nodes = len(S)

# PageRank damping/teleportation factor
d = 0.85

# Initial pagerank
pagerank_initial = {}
for link in crawled_links:
    pagerank_initial[link] = 1/all_pages

import math
count = 0 
i = 0
perplexity_c = []
while count < 4: 
    # while loop
    sinkPR = 0
    # calculate total sink PR 
    for link in S:                
        sinkPR += pagerank_initial[link]
    newpr = {}
    for link in crawled_links:
        newpr[link] = (1-d)/all_pages
        newpr[link] += d*sinkPR/all_pages
        for l in link_inlinks[link]:
            newpr[link] += d*pagerank_initial[l]/(len(crawled_outlinks[l]))
    for link in crawled_links:
        pagerank_initial[link] = newpr[link]

    # Calculating Entorpy
    entropy = 0
    for link in crawled_links:
        entropy += (-1)*(pagerank_initial[link]* math.log(pagerank_initial[link],2))
    
    # Calculating Perplexity
    perplexity_c.append(math.pow(2,entropy))   
       
    if(len(perplexity_c) > 0):
        if(abs(perplexity_c[i]-perplexity_c[i-1]) < 1):
            count = count + 1
        else:
            count = 0

    i = i + 1



# Sorting documents based on pagerank
sorted_links = [[j, v] for j, v in pagerank_initial.items()]
sorted_links.sort(key=itemgetter(1), reverse=True)


#------------------------------------------------------------------------------
# WT2G
#------------------------------------------------------------------------------


# In links
file = open('/Users/satishreddychirra/Document/wt2g_inlinks.txt', 'r')
wt2g_inlinks = {}
for line in file:
    wt2g_inlinks[line.split(' ')[0].strip('\n')] = line.split(' ')[1:-1]


# Out Links
outlinks = []
for key in wt2g_inlinks:
    for link in set(wt2g_inlinks[key]):        
        outlinks.append((link, key))
        
from collections import defaultdict
wt2g_outlinks = defaultdict(list)

for p, v in outlinks:
    wt2g_outlinks[p].append(v)

# Sink Links
s_links = []
for key in wt2g_inlinks:
    if(key not in wt2g_outlinks.keys()):
        s_links.append(key)

# Pagerank
N = len(wt2g_inlinks)
sink_nodes = N-len(wt2g_outlinks)

# PageRank damping/teleportation factor
d = 0.85

# Total Links
links_c = []
for key in wt2g_inlinks:
    links_c.append(key)

# Outlink Lenght
outlink_len = {}
for key in wt2g_outlinks:
    outlink_len[key] = len(wt2g_outlinks[key])


# Initial pagerank
pagerank = {}
for link in links_c:
    pagerank[link] = 1/N



import math
count = 0 
i = 0
perplexity = []
while count < 4: 
    sinkPR = 0
    # calculate total sink PR 
    for link in s_links:                
        sinkPR += pagerank[link]
    new_pr = {}
    for link in links_c:
        new_pr[link] = (1-d)/N
        new_pr[link] += d*sinkPR/N
        for l in set(wt2g_inlinks[link]):
            new_pr[link] += d*pagerank[l]/outlink_len[l]
    for link in links_c:
        pagerank[link] = new_pr[link]

    # Calculating Entorpy    
    entropy = 0
    for link in links_c:
        entropy += (-1)*(pagerank[link]* math.log(pagerank[link],2))       
    # Calculating Perplexity
    perplexity.append(2**entropy)       
    
    if(len(perplexity) > 0):
        if(abs(perplexity[i]-perplexity[i-1]) < 1):
            count = count + 1
        else:
            count = 0

    i = i + 1
    

# Sorting documents based on pagerank
sorted_pr = [[j, v] for j, v in pagerank.items()]
sorted_pr.sort(key=itemgetter(1), reverse=True)

#------------------------------------------------------------------------------
# HITS
#------------------------------------------------------------------------------

# BM25 
import re
import elasticsearch.helpers
from collections import Counter
D = len(crawled_links)
results = {}

query_file = open('/Users/satishreddychirra/Document/query_A4.txt')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    qry_score = Counter()
    for word in qry['Query Terms']:
        print(word)
        es = Elasticsearch(timeout = 30)
        res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match": {"text": word}},
                              "script_fields": {"index_df": {"script": {"lang": "groovy","inline": "_index['text']['" + word + "'].df()"}},
                              "index_tf": {"script": {"lang": "groovy","inline": "_index['text']['" + word + "'].tf()"}}}},
                              index="team_asr", doc_type="document", scroll=u"5m")
        
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
            bm25_score.append((math.log((D+0.5)/(doc_freq[i]+0.5))) * ((tf[i] + 1.2*tf[i])/(tf[i] + 1.2*((1-0.75)+ 0.75 * (doc_len[i]/avg_doc_len)))))

        results.clear()
        results = dict(zip(doc_no, bm25_score))

        qry_score = Counter(results) + qry_score

    score = dict(qry_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)

    rank = list(range(1, 1001))

    results_file = open('/Users/satishreddychirra/Document/bm25_results_A4.txt', 'a')
    for i in range(1000):
        results_file.writelines( str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')
    results_file.close()


# Creating root set from the top 1000 links of BM25
root_links = []
for i in range(1000):
    root_links.append(score_list[i][0])

# Getting Out links using the root set
import random
links_final = []
for link in root_links:
    t = []
    for l in crawled_outlinks[link]:
        t.append(l)
        random.shuffle(t)
    links_final.append(t)

# Getting In links using the root set
for link in root_links:
    t = []
    for l in link_inlinks[link]:
        t.append(l)   
        random.shuffle(t)
    links_final.append(t)

# Creating a base set of approx 10000 links
base_links = []
for link in root_links:
    base_links.append(link)

for i in range(len(links_final)):
    for link in links_final[i]:
        if(len(base_links) < 10000 and link in crawled_links and link not in base_links):
            base_links.append(link)


base_set = {}
for link in base_links:
    base_set[link] = 1

# Hub & Authority

hub_score = {}
auth_score = {}
for link in base_links:
    hub_score[link] = 1
    auth_score[link] = 1

norm_h = 0 
norm_a = 0
count = 0
i = 0
normalized_a = []
normalized_h = []

while(count < 4): 
    
    norm_a = 0
    for link in base_links:
        auth_score[link] = 0
        for l in link_inlinks[link]:
            if(l in base_set.keys()):
                auth_score[link] += hub_score[l]
        norm_a += (auth_score[link]**2)
    norm_a = math.sqrt(norm_a)    
    
    for link in base_links:
        auth_score[link] = auth_score[link]/norm_a
    
    norm_h = 0
    for link in base_links:
        hub_score[link] = 0
        for l in crawled_outlinks[link]:
            if(l in base_set.keys()):
                hub_score[link] += auth_score[l]
        norm_h += (hub_score[link]**2)
    norm_h = math.sqrt(norm_h)      
    
    for link in base_links:
        hub_score[link] = hub_score[link]/norm_h    
    
    normalized_a.append(norm_a)
    normalized_h.append(norm_h)
       
    if(len(normalized_a) > 0 and len(normalized_h) > 0):
        if(abs(normalized_a[i]-normalized_a[i-1]) < 1 and abs(normalized_h[i]-normalized_h[i-1]) < 1):
            count = count+ 1
        else:
            count = 0
      
    i = i + 1
