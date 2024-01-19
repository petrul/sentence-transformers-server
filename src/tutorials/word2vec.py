#! /usr/bin/env python3

from gensim.models import Word2Vec

sentences = [
    ["cat", "say", "meow"], 
    ["dog", "say", "woof"]
];

model = Word2Vec(sentences, min_count=1)
print(model.wv['cat'])
print(model.wv['cat', 'say'])
# print(model["cat"])