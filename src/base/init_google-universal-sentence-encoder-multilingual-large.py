import tensorflow_hub as hub
import tensorflow_text

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
sentences = ["This is an example sentence.", "Here is another sentence."]
embeddings = embed(sentences)
print(embeddings)