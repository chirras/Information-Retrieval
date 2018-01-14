# Information-Retrieval
Course Assignments


## 1

Implemented and compared various retrieval systems using vector space models (Okapi TF, TF-IDF, Okapi BM25) 
and language models (Laplace smoothing, Jelinek-Mercer smoothing).

It has,
A program to parse the corpus and index it with elasticsearch
A query processor, which runs queries from an input file using a selected retrieval model


## 2

Implemented own index to take the place of elasticsearch in the previous code, and index the document collection used previously. 

This involves two programs,

A tokenizer and indexer
An updated version of A1 ranker which uses my own inverted index


## 3

A web crawler that crawl Internet documents to construct a document collection focused on a particular topic which
conform strictly to politeness policy. 


## 4

Web graph computation - Computed link graph measures for each page crawled using the adjacency matrix.

Computed the PageRank of every page in crawled in previous assignment and Listed the top 500 pages by the PageRank score. Also
computed Hubs and Authority score for the pages in the crawl and Created files for top 500 hub webpages and top 500 authority webpages.


## 5 

Evaluated the previous assignment using vertical search engine. Manual assessments have been given, using your vertical search engine and a web interface
on a scale of “non-relevant”, “relevant”, “very relevant”. And also rewrote treceval.


## 6

In this assignment, represented documents as numerical features, and apply machine learning to obtain retrieval ranked lists using the data from Assignment 1.


## 7

Built a Spam Classifier using Machine Learning and ElasticSearch.


## 8

Clustered documents, detected topics and represented documents in topic space.






