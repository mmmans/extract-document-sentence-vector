# #####################################################################################
# -*- coding:utf-8 -*-
# author :xin_men
# description : training Doc2vec model to extract the vector of sentence/articles
# and classifer model to classify the emotion of the articles/sentences
# The code use IMDB(http://ai.stanford.edu/~amaas/data/sentiment/) dataset, the 
# preprocessd data is provide with this code at (https://github.com/mans-men/extract-document-sentence-vector) 
# ##################################################################################### 

import numpy as np
import gensim
import os
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from sklearn.model_selection import train_test_split 
import random

def get_dataset(pos_file,neg_file,unsup_file):
    ''' read data from files
    Args:
        param pos_file: the positive reviews of movies, one line represent one piece of review
        param neg_file; the negative reviews of movies, one line represent one piece of review
        param unsup_file: the reviews that is neither positive nor negative,one line represent one piece of review
    Returns:
        x_train: train data list, a list of tagged documents
        x_test:  test data list, a list of tagged documents
        unsup_reviews:  a list of tagged documents,which is has not labels
        y_train: train data labels
        y_test: train data labels
    '''
    #read data
    with open(pos_file,'r',encoding="utf-8") as infile:
        pos_reviews = infile.readlines()
    with open(neg_file,'r',encoding="utf-8") as infile:
        neg_reviews = infile.readlines()
    with open(unsup_file,'r',encoding="utf-8") as infile:
        unsup_reviews = infile.readlines()

    # 1 denotes positive emotion，0 denotes negative emotion
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    # split the test and train dataset
    x = np.concatenate((pos_reviews,neg_reviews))
    #x = pos_reviews + neg_reviews
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # text clean for english
    def cleanText(corpus):
        punctuation = """.,?!:;(){}[]"""
        corpus = [z.lower().replace('\n','') for z in corpus]
        corpus = [z.replace('<br />', ' ') for z in corpus]

        #treat punctuation as individual words
        for c in punctuation:
            corpus = [z.replace(c, ' %s '%c) for z in corpus]
        corpus = [z.split() for z in corpus]
        return corpus

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    unsup_reviews = cleanText(unsup_reviews)

    # Gensim.model.Doc2Vec requires that every aritcles/sentences labeled a unique label
    # this could be done the TaggedDocument API.
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '{0}_{1}'.format(label_type,i)
            labelized.append(TaggedDocument(v, tags=[label]))
        return labelized

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')
    return x_train,x_test,unsup_reviews,y_train, y_test

def getVecs(model, corpus):
    '''obtain the vectors of a list of documents
    Args:
        model: the model which we used to extract the vector
        corpus: a list of tagged document
    Returns:
        vecs : the vectors that extract from corpus
    '''
    #vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    vecs  = [model.infer_vector(z.words).reshape((1,-1)) for z in corpus]
    return np.concatenate(vecs)

def train(x_train,x_test,unsup_reviews,size = 400,epoch_num=10):
    '''train and reture the model
    initialzie two models, we will concatenate the vectors extract from them when predict
    for more details, please refer to https://radimrehurek.com/gensim/models/doc2vec.html
    Args:
        x_train:  train data list, a list of tagged documents
        x_test: test data list, a list of tagged documents
        unsup_reviews: a list of tagged documents,which is has not labels
        size : int, indicates the vector length we want to obtain
        epoch_nums : epochs to train the model
    Returns:
        model_dm : Distributed Memory version of Paragraph Vector model
        model_dbow : Bag of Words version of Paragraph Vector (PV-DBOW) model
    '''

    model_dm = Doc2Vec(document = x_train,min_count=1, window=10, vector_size=size, sample=1e-3, negative=5, workers=8,epochs=10)
    model_dbow = Doc2Vec(document = x_train,min_count=1, window=10, vector_size=size, sample=1e-3, negative=5, dm=0, workers=8,epochs=10)
    # build vocabulary
    vocab = list(x_train)
    vocab += x_test
    vocab += unsup_reviews
    model_dm.build_vocab(vocab)
    model_dbow.build_vocab(vocab)
    # train the model with the train data
    all_train_reviews = x_train+unsup_reviews 
    for epoch in range(epoch_num):
        print("epoch :{0}".format(epoch))
        random.shuffle(all_train_reviews)
        model_dm.train(all_train_reviews,total_examples=model_dm.corpus_count,epochs=model_dm.epochs)
        model_dbow.train(all_train_reviews,total_examples=model_dbow.corpus_count,epochs=model_dbow.epochs)
    return model_dm,model_dbow
 
def get_vectors(model_dm,model_dbow,x_train,x_test):
    '''transform all documents to vectors
    Args:
        model_dm:  Distributed Memory version of Paragraph Vector model
        model_dbow: Bag of Words version of Paragraph Vector (PV-DBOW) model
        x_train:train data list, a list of tagged documents
        x_test:test data list, a list of tagged documents
    Returns:
        train_vecs : the vectors that extract from training data
        test_vecs : the vectors that extract from test data
    '''

    # transform train data
    train_vecs_dm = getVecs(model_dm, x_train)
    train_vecs_dbow = getVecs(model_dbow, x_train)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    # transform test data
    test_vecs_dm = getVecs(model_dm, x_test)
    test_vecs_dbow = getVecs(model_dbow, x_test)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
    return train_vecs,test_vecs

def Classifier(train_vecs,y_train,test_vecs, y_test):
    '''train a classifer with vecs and its labels
    Args:
        train_vecs:  vectors extracted from training data
        y_train: training data label
        test_vecs: vectors extracted from test data
        y_test : test data label
    Returns:
        lr : the trained classifier model
    '''
    from sklearn.linear_model import SGDClassifier
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)
    print ('Test Accuracy: {0}'.format(lr.score(test_vecs, y_test)))
    return lr

def ROC_curve(lr,x_test,y_test):
    '''plot the ROC curve
    Args:
        lr:  the trained classifier model
        x_test: vecs extract from test data
        y_test : test data label
    Returns:
        null
    '''
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    pred_probas = lr.predict_proba(x_test)[:,1]
    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()

if __name__ == "__main__":
    
    root_path = "./data"
    x = os.path.join(root_path,"pos.txt")
    y = os.path.join(root_path,"neg.txt")
    z = os.path.join(root_path,"unsup.txt")
    vector_length,epoch_num = 200,1
    x_train,x_test,unsup_reviews,y_train, y_test = get_dataset(x,y,z)
    model_dm,model_dbow = train(x_train,x_test,unsup_reviews,vector_length,epoch_num)
    train_vecs,test_vecs = get_vectors(model_dm,model_dbow,x_train,x_test)
    lr=Classifier(train_vecs,y_train,test_vecs, y_test)
    ROC_curve(lr,test_vecs,y_test) 