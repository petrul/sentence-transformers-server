#! /usr/bin/python3
from towhee import AutoPipes
p = AutoPipes.pipeline('sentence_embedding')
# output = p('Hello World.').get()
output = p('Foaie verde, foaie lată.').get()
print(output)
