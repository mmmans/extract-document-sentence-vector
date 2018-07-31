# extract-document-sentence-vector
This project trained a a Doc2vec model, which could extract vectors from articles/sentence. The vectors could be used to calculate the similarity of two articles, or classfiy the articles. for more doc2vec details and usage, please refer to [https://radimrehurek.com/gensim/models/doc2vec.html](https://radimrehurek.com/gensim/models/doc2vec.html)

The project trained the Doc2vec model with [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/) dataset, then used the trained model to extract the vectors of the movie reviews which is labeled either positive or negtive. Finally, we introduce the Classifier trained on the vectors and its label. Ther result shows that the classifier obtained an accuracy of 86+%. 
 ![sample](https://github.com/mans-men/extract-document-sentence-vector/blob/master/result.png)
 ![enen](https://github.com/mans-men/extract-document-sentence-vector/blob/master/sample.png)

## Environment
* win10
* python 3.6
* Anaconda3
* gensim 3.4.0

## steps to run
* clone the project
* open the terminal 
* python main.py

