#! /usr/bin/env python3
# 
# from
# https://www.sbert.net/docs/usage/semantic_textual_similarity.html
# 
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

# Two lists of sentences
sentences1 = ['The cat sits outside',
             'A man is playing guitar',
             'The new movie is awesome',
             'A woman wants to have sex.']

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great',
              'A boy masturbates.',
              "C'est la vie"]

# print (sentences1[0])
# print(util.cos_sim(sentences1[0], sentences1[0]))

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
print (embeddings1[0].size())

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)


# print(cosine_scores)
#Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))