# OKAPI TF Score Calculation

global doc_score, doc_no, tf_nor, tf, tf_nor

results = {}


import re
from nltk.corpus import stopwords

query_file = open('/Users/satishreddychirra/Downloads/AP_DATA/query_desc.51-100.short.txt')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    from collections import Counter
    qry_score = Counter()
    for word in qry['Query Terms']:
        if word not in (
                stopwords.words('english'), 'document', 'identify', 'will', 'must', 'the', 'of', 'or', 'a', 'on', 'in',
                'an', 'about', 'to', 'by', 'at', 'into', 'one', 'u', 's', 'and', 'with', 'as', 'which', 'any', 'has', 'd',
                'its', 'how', 'mci', 'describe', 'either'):

            from nltk.stem import PorterStemmer, WordNetLemmatizer

            stemmer = PorterStemmer()
            lemmatiser = WordNetLemmatizer()
            word_s = ("%s" % (stemmer.stem(word)))
            from elasticsearch import Elasticsearch
            import elasticsearch.helpers

            es = Elasticsearch()
            res = elasticsearch.helpers.scan(es,
                                             {"_source": True, "query": {"match": {"text": word_s}},
                                              "script_fields": {"index_df": {"script": {"lang": "groovy",
                                                                                        "inline": "_index['text']['" + word_s + "'].df()"}},
                                                                "index_tf": {"script": {"lang": "groovy",
                                                                                        "inline": "_index['text']['" + word_s + "'].tf()"}}}},
                                             index="ap_dataset", doc_type="document", scroll=u"5m")

            res_list = []
            for i in res:
                res_list.append(i)


            tf = []
            doc_no = []
            doc_len = []
            tf_nor = []
            for i in range(len(res_list)):
                tf.append(res_list[i]['fields']['index_tf'][0])
                doc_no.append(res_list[i]['_id'])
                doc_len.append(res_list[i]['_source']['length'])
                tf_nor.append(tf[i] / ((tf[i] + 0.5 + 1.5 * (doc_len[i] / 441))))

            results.clear()
            results = dict(zip(doc_no, tf_nor))



            qry_score = Counter(results) + qry_score




    from operator import itemgetter

    score = dict(qry_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)

    rank = list(range(1, 1001))

    results_file = open('/Users/satishreddychirra/Document/okapitf_results.txt', 'a')
    for i in range(1000):
        results_file.writelines( str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# TF-IDF Calculation for all queries
global doc_score, doc_no, tf_nor, tf, tf_nor

results = {}
D = 84678
import math
import re
from nltk.corpus import stopwords

query_file = open('/Users/satishreddychirra/Downloads/AP_DATA/query_desc.51-100.short.txt')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    from collections import Counter
    qry_score = Counter()
    for word in qry['Query Terms']:
        if word not in (
                stopwords.words('english'), 'document', 'identify', 'will', 'must', 'the', 'of', 'or', 'a', 'on', 'in',
                'an', 'about', 'to', 'by', 'at', 'into', 'one', 'u', 's', 'and', 'with', 'as', 'which', 'any', 'has', 'd',
                'its', 'how', 'mci', 'describe', 'either'):

            from nltk.stem import PorterStemmer, WordNetLemmatizer

            stemmer = PorterStemmer()
            lemmatiser = WordNetLemmatizer()
            word_s = ("%s" % (stemmer.stem(word)))
            from elasticsearch import Elasticsearch
            import elasticsearch.helpers

            es = Elasticsearch()
            res = elasticsearch.helpers.scan(es,
                                             {"_source": True, "query": {"match": {"text": word_s}},
                                              "script_fields": {"index_df": {"script": {"lang": "groovy",
                                                                                        "inline": "_index['text']['" + word_s + "'].df()"}},
                                                                "index_tf": {"script": {"lang": "groovy",
                                                                                        "inline": "_index['text']['" + word_s + "'].tf()"}}}},
                                             index="ap_dataset", doc_type="document", scroll=u"5m")

            res_list = []
            for i in res:
                res_list.append(i)


            tf = []
            doc_no = []
            doc_len = []
            tf_nor = []
            doc_freq = []
            for i in range(len(res_list)):
                tf.append(res_list[i]['fields']['index_tf'][0])
                doc_no.append(res_list[i]['_id'])
                doc_len.append(res_list[i]['_source']['length'])
                doc_freq.append(res_list[i]['fields']['index_df'][0])
                tf_nor.append((tf[i] / (tf[i] + 0.5 + 1.5 * (doc_len[i] / 441))) * math.log(D/doc_freq[i]))

            results.clear()
            results = dict(zip(doc_no, tf_nor))



            qry_score = Counter(results) + qry_score




    from operator import itemgetter

    score = dict(qry_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)

    rank = list(range(1, 1001))

    results_file = open('/Users/satishreddychirra/Document/tfidf_results.txt', 'a')
    for i in range(1000):
        results_file.writelines( str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# BM25 Score Calculation for all queries

global doc_score, doc_no, tf_nor, tf, tf_nor

results = {}
D = 84678
import math
import re
from nltk.corpus import stopwords

query_file = open('/Users/satishreddychirra/Downloads/AP_DATA/query_desc.51-100.short.txt')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}
    from collections import Counter
    qry_score = Counter()
    for word in qry['Query Terms']:
        if word not in (
                stopwords.words('english'), 'document', 'identify', 'will', 'must', 'the', 'of', 'or', 'a', 'on', 'in',
                'an', 'about', 'to', 'by', 'at', 'into', 'one', 'u', 's', 'and', 'with', 'as', 'which', 'any', 'has', 'd',
                'its', 'how', 'mci', 'describe', 'either'):

            from nltk.stem import PorterStemmer, WordNetLemmatizer

            stemmer = PorterStemmer()
            lemmatiser = WordNetLemmatizer()
            word_s = ("%s" % (stemmer.stem(word)))
            from elasticsearch import Elasticsearch
            import elasticsearch.helpers

            es = Elasticsearch()
            res = elasticsearch.helpers.scan(es,
                                             {"_source": True, "query": {"match": {"text": word_s}},
                                              "script_fields": {"index_df": {"script": {"lang": "groovy",
                                                                                        "inline": "_index['text']['" + word_s + "'].df()"}},
                                                                "index_tf": {"script": {"lang": "groovy",
                                                                                        "inline": "_index['text']['" + word_s + "'].tf()"}}}},
                                             index="ap_dataset", doc_type="document", scroll=u"5m")

            res_list = []
            for i in res:
                res_list.append(i)


            tf = []
            doc_no = []
            doc_len = []
            tf_nor = []
            doc_freq = []
            for i in range(len(res_list)):
                tf.append(res_list[i]['fields']['index_tf'][0])
                doc_no.append(res_list[i]['_id'])
                doc_len.append(res_list[i]['_source']['length'])
                doc_freq.append(res_list[i]['fields']['index_df'][0])
                tf_nor.append((math.log((D+0.5)/(doc_freq[i]+0.5))) * ((tf[i] + 1.2*tf[i])/(tf[i] + 1.2*((1-0.75)+ 0.75 * (doc_len[i]/441)))))


            results.clear()
            results = dict(zip(doc_no, tf_nor))



            qry_score = Counter(results) + qry_score




    from operator import itemgetter

    score = dict(qry_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)

    rank = list(range(1, 1001))

    results_file = open('/Users/satishreddychirra/Document/bm25_results.txt', 'a')
    for i in range(1000):
        results_file.writelines( str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')





















































