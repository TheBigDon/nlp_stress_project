import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import matplotlib.pyplot as plt

with open("Data/normal_form_messages.txt") as f:
    document = f.read()

bag_words = document.split(' ')
unique_words = set(bag_words)

amount_words = dict.fromkeys(unique_words, 0)
for word in bag_words:
    amount_words[word] += 1

nltk.download('stopwords')

stopwords_list = stopwords.words('russian')
[amount_words.pop(key, None) for key in stopwords_list]


def compute_tf(word_dict, bag_words):
    tf_dict = {}
    bag_count_words = len(bag_words)
    for word, count in word_dict.items():
        tf_dict[word] = count/float(bag_count_words)
    return tf_dict


tf_document = compute_tf(amount_words, bag_words)
sorted_tf_document = dict(sorted(tf_document.items(), key=itemgetter(1)))
list_words_key = list(sorted_tf_document.keys())
list_words_value = list(sorted_tf_document.values())

plt.figure(figsize=(15, 4))
plt.stem(list_words_key[-11:], list_words_value[-11:])
plt.show()


def compute_idf(documents):
    import math
    N = len(documents)

    idf_dict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idf_dict[word] += 1

    for word, val in idf_dict.items():
        idf_dict[word] = math.log(N / float(val))
    return idf_dict


idf = compute_idf([amount_words, amount_words])


def compute_tfidf(tf_bag_words, idf):
    tfidf = {}
    for word, val in tf_bag_words.items():
        tfidf[word] = val * idf[word]
    return tfidf


tfidf = compute_tfidf(tf_document, idf)
df = pd.DataFrame([tfidf])

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([document])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)