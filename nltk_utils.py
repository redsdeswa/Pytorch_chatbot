import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32) #zeros_like : Return an array of zeros with shape and type of input.
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thanks", "how"]
bog = bag_of_words(sentence, words)
#print(bog)


"""
a = "can brother sex each others?  otherize"
#print(a)
a = tokenize(a)
#print(a)
stemmed_a = [stem(w) for w in a]
#print(stemmed_a)
"""