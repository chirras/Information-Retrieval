# Laplace Calculation

from elasticsearch import Elasticsearch
es = Elasticsearch()

global doc_score, doc_no, tf_nor, tf, vocSize

import math
from collections import Counter
import re
from nltk.corpus import stopwords

import math

def p_laplace(term):
    resu_list = []
    dict_t = {}
    import elasticsearch.helpers
    resu = elasticsearch.helpers.scan(es, query={"_source": True, "query": {"match": {"text": str(term)}}}, index='ap_dataset', scroll=u'1m', doc_type='document')
    for i in resu:
        resu_list.append(i)

    voc_size = es.search(index="ap_dataset", body={"aggs": {"vocabSize": {"cardinality": {"field": "text"}}}})
    vocSize = voc_size['aggregations']['vocabSize']['value']

    for i in range(0, len(resu_list)):
        doc_id = resu_list[i]['_id']
        doc_l = resu_list[i]['_source']['length']
        dic = {doc_id: ((1/(doc_l + vocSize)))}
        dict_t.update(dic)
    return (dict_t)



def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


query_file = open('/Users/satishreddychirra/Document/query')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}

    ts = p_laplace(qry['Query Terms'])
    document_no = []
    for key, value in ts.items():
        document_no.append(key)

    score = zerolistmaker(len(document_no))
    results = dict(zip(document_no, score))

    temp_score = Counter(results)
    total_score = Counter(results)

    for word in qry['Query Terms']:
        if word not in (stopwords.words('english')):

            from nltk.stem import PorterStemmer, WordNetLemmatizer

            stemmer = PorterStemmer()
            lemmatiser = WordNetLemmatizer()
            word_s = ("%s" % (stemmer.stem(word)))
            from elasticsearch import Elasticsearch
            import elasticsearch.helpers

            es = Elasticsearch()
            res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match": {"text": word_s}},
                                          "script_fields": {"index_df": {"script": {"lang": "groovy",
                                                                                    "inline": "_index['text']['" + word_s + "'].df()"}},
                                                            "index_tf": {"script": {"lang": "groovy",
                                                                                    "inline": "_index['text']['" + word_s + "'].tf()"}}}},
                                         index="ap_dataset", doc_type="document", scroll=u"5m")

            res_list = []
            for i in res:
                res_list.append(i)

            voc_size = es.search(index="ap_dataset", body={"aggs": {"vocabSize": {"cardinality": {"field": "text"}}}})
            vocSize = voc_size['aggregations']['vocabSize']['value']

            tf = []
            doc_no = []
            doc_len = []
            lm_laplace = []
            for i in range(len(res_list)):
                tf.append(res_list[i]['fields']['index_tf'][0])
                doc_no.append(res_list[i]['_id'])
                doc_len.append(res_list[i]['_source']['length'])
                lm_laplace.append(((tf[i] + 1 )/ (doc_len[i] + vocSize)))

            lm_score = Counter(dict(zip(doc_no, lm_laplace)))
            p_score = Counter(p_laplace(qry['Query Terms']))

            score_t = p_score.copy()
            score_t.update(lm_score)


            for p, v in score_t.items():
                score_t.update({p: math.log(v)})

            total_score.update(score_t)

    from operator import itemgetter

    score = dict(total_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)

    rank = list(range(1, 1001))
    results_file = open('/Users/satishreddychirra/Document/laplace_results.txt', 'a')
    for i in range(1000):
        results_file.writelines(
            str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(
                score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# JM Smoothing

from elasticsearch import Elasticsearch

es = Elasticsearch()

global doc_score, doc_no, tf_nor, tf, vocSize

import math
from collections import Counter
import re
from nltk.corpus import stopwords

import math


def jm_dscore(term):
    resu_list = []
    dict_t = {}
    import elasticsearch.helpers
    resu = elasticsearch.helpers.scan(es, query={"_source": True, "query": {"match": {"text": str(term)}}},
                                      index='ap_dataset', scroll=u'1m', doc_type='document')
    for i in resu:
        resu_list.append(i)

    voc_size = es.search(index="ap_dataset", body={"aggs": {"vocabSize": {"cardinality": {"field": "text"}}}})
    vocSize = voc_size['aggregations']['vocabSize']['value']

    for i in range(0, len(resu_list)):
        doc_id = resu_list[i]['_id']
        dic = {doc_id: (0.0)}
        dict_t.update(dic)
    return (dict_t)


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


query_file = open('/Users/satishreddychirra/Document/query')
for line in query_file.readlines():
    query = line.strip()
    sen = query.lower()
    words = re.findall(r'\w+', sen, flags=re.UNICODE)
    qry = {"Query No": words[0], "Query Terms": words[1:]}

    ts = jm_dscore(qry['Query Terms'])
    document_no = []
    for key, value in ts.items():
        document_no.append(key)

    score = zerolistmaker(len(document_no))
    results = dict(zip(document_no, score))

    temp_score = Counter(results)
    total_score = Counter(results)

    for word in qry['Query Terms']:
        if word not in (stopwords.words('english')):

            from nltk.stem import PorterStemmer, WordNetLemmatizer

            stemmer = PorterStemmer()
            lemmatiser = WordNetLemmatizer()
            word_s = ("%s" % (stemmer.stem(word)))
            from elasticsearch import Elasticsearch
            import elasticsearch.helpers

            es = Elasticsearch()
            res = elasticsearch.helpers.scan(es, {"_source": True, "query": {"match": {"text": word_s}},
                                                  "script_fields": {"index_ttf": {"script": {"lang": "groovy",
                                                                                            "inline": "_index['text']['" + word_s + "'].ttf()"}},
                                                                    "index_tf": {"script": {"lang": "groovy",
                                                                                            "inline": "_index['text']['" + word_s + "'].tf()"}}}},
                                             index="ap_dataset", doc_type="document", scroll=u"5m")

            res_list = []
            for i in res:
                res_list.append(i)

            voc_size = es.search(index="ap_dataset", body={"aggs": {"vocabSize": {"cardinality": {"field": "text"}}}})
            vocSize = voc_size['aggregations']['vocabSize']['value']

            tf = []
            doc_no = []
            doc_len = []
            cf = []
            jm_smooth = []
            for i in range(len(res_list)):
                tf.append(res_list[i]['fields']['index_tf'][0])
                doc_no.append(res_list[i]['_id'])
                doc_len.append(res_list[i]['_source']['length'])
                cf.append(res_list[i]['fields']['index_ttf'][0])
                jm_smooth.append(((0.8)*(tf[i]/doc_len[i])) + ((0.2) * (cf[i]/vocSize)))

            jm_score = Counter(dict(zip(doc_no, jm_smooth)))
            jm_d = Counter(jm_dscore(qry['Query Terms']))

            for key, value in jm_d.items():
                jm_d.update({key: (value+((0.2) * (cf[0]/vocSize)))})

            score_t = jm_d.copy()
            score_t.update(jm_score)


            for p, v in score_t.items():
                score_t.update({p: math.log(v)})

            total_score.update(score_t)

    from operator import itemgetter

    score = dict(total_score)
    score_list = [[i, v] for i, v in score.items()]
    score_list.sort(key=itemgetter(1), reverse=True)

    rank = list(range(1, 1001))
    results_file = open('/Users/satishreddychirra/Document/jm_smooth_results.txt', 'a')
    for i in range(1000):
        results_file.writelines(
            str(qry['Query No']) + " " + str("Q0") + " " + str(score_list[i][0]) + " " + str(rank[i]) + " " + str(
                score_list[i][1]) + " " + str("Exp"))
        results_file.write('\n')
