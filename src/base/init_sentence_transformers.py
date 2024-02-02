#! /usr/bin/env python3
# 
# from
# https://www.sbert.net/docs/usage/semantic_textual_similarity.html
# 
from sentence_transformers import SentenceTransformer

modelnames = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
print(f'will download {len(modelnames)} models: {modelnames}')

for name in modelnames:
    print(f'downloading {name}')
    model = SentenceTransformer(name)

    # Two lists of sentences
    sentences1 = ['The cat sits outside',
                'A man is playing guitar',
                'The new movie is awesome',
                'A woman wants to have sex.']

    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    print (embeddings1)
