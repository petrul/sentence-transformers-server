#! /usr/bin/python3

import spacy
# Load the pre-trained word embeddings from spacy
nlp = spacy.load("en_core_web_md")

# Define a sample text containing named entities
text = "Barack Obama was the 44th President of the United States."# Use spacy to identify the named entities in the text
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
