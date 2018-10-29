from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import numpy as np

min_count = 1
window = 10
vector_size = 100
negative = 5
workers = 2
epochs = 50
dbow_words = 0


class Doc_vectors(object):
    def __init__(self):
        self.min_count = min_count
        self.window = window
        self.vector_size = vector_size
        self.negative = negative
        self.workers = workers
        self.epochs = epochs
        self.dbow_words=dbow_words
        
    #把文本数据转成doc2vec要求的格式，即生成words
    def list2tag(self, corpus):
        documents = []
        for i, text in enumerate(list(corpus)):
            words_list = text.split(' ')
            documents.append(TaggedDocument(words=words_list, tags=[i]))
        return documents

    #建立doc2vec模型，并获取doc_vectors      
    def build_doc_model(self, documents, dm):
        model = Doc2Vec(min_count=self.min_count, window=self.window, vector_size=self.vector_size, negative=self.negative, workers=self.workers, epochs=self.epochs, dm=dm, dbow_words=self.dbow_words)
        #建立词典
        model.build_vocab(documents)
        #训练模型
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    #获取doc_vectors
    def getVecs(self, model, documents):
        vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, self.vector_size)) for z in documents]
        return np.concatenate(vecs)