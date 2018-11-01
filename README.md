# text-similarity

all kinds of baseline models for modeling tasks with pair of sentences: semantic text similarity(STS), natural language inference(NLI), paraphrase identification(PI), question answering(QA).

1.Desc
this repository contain models that learn to detect sentence similarity for natural language understanding tasks.

there are two different kinds of models:

sentence encoding-based models that separate the encoding of the individual sentences,

joint methods that allow to use encoding of both sentences( to use cross-features or attention from one sentence to the other)

we will try to cover both of these two methods.

find more about task, data or even start AI completation by check here:

https://dc.cloud.alipay.com/index#/topic/data?id=3

2.Data Processing: data enhancement and word segmentation strategy
length of sentence. 5 stand for less than 5; 10 stand for great than 5 and less than 10

source data in .csv file.

data format: line_no,sentence1,sentence2,label. 4 columns are splitted by "\t"

 001\t question1\t question2\t label
{ 5: 0.11388705332181162, 10: 0.6559243633406191, 15: 0.1654043613073756, 20: 0.04325725613785391})

as you can see that most of sentences in this task is quite short, short less 15 or 20.

a.swap sentence 1 and sentence 2

   if sentence 1 and sentence 2 represent the same meaning, then sentence 2 and sentence 1 also have same meaning.

   check: method get_training_data() at data_util.py
b.randomly change order given a sentence.

   as same key words in the same may contain most important message in a sentence, change order of these key words should also able to send those message;

   however there may exist cases, which it not count that big percentage, that meaning of sentence way changed when we change order of words.

   check: method get_training_data() at data_util.py
after data enhancement:length of training data: 81922 ;validation data: 1600; test data:800; percent of true label: 0.217

c.tokenize style

 you can train the model use character, or word or pinyin. for example even you train this model in pinyin, it still can get pretty reasonable performance.

 tokenize sentence in pinyin: we will first tokenize sentence into word, then translate it into pinyin. e.g. it now become: ['nihao', 'wo', 'de', 'pengyou']
3.Feature Engineering
get data mining features given two sentences as string.

1)n-gram similiarity(blue score for n-gram=1,2,3...);

2) get length of questions, difference of length

3) how many words are same, how many words are unique

4) question 1,2 start with how/why/when(wei shen me,zenme，ruhe，weihe）

5）edit distance

6) cos similiarity using bag of words for sentence representation(combine tfidf with word embedding from word2vec,fasttext)

7) manhattan_distance,canberra_distance,minkowski_distance,euclidean_distance
check data_mining_features method under data_util.py

4.Imbalance Classification for Skew Data
20% percent is postive label, 80% is negative label. by predict negative for all test data, you can get 80% acc, but recall is 0%.

1)if you random guess, what is f1 score for your task?

by using random number+ feed forward(fully connected) layer, f1 score is around 0.34. so after lots of work and if your model achive less then 0.4,

obviously it is not a good model.

how to adjust the weight for each label?
one way is to calculate validate accuracy for each label after a epoch, and use it as indicator to adjust the weight. set high weight for label with low accuracy.

but for small dataset, validate accuracy may fluctuate(unstable), so you can use move average of accuracy or set a ceiling value for the weight.

check weight_boosting.py
