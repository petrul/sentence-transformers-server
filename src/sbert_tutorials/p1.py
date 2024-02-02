#! /usr/bin/env python3

# from a milvus link, several ways of computing similarity and comparation. 

from sentence_transformers import SentenceTransformer
from milvus import default_server
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from time import time

# default_server.start()
# connections.connect(host="127.0.0.1", port=default_server.listen_port)
# connections.connect(host="127.0.0.1", port=9091)

v12 = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ft3_v12 = SentenceTransformer("Sprylab/paraphrase-multilingual-MiniLM-L12-v2-fine-tuned-3")
ft5_v12 = SentenceTransformer("hroth/psais-paraphrase-multilingual-MiniLM-L12-v2-5shot")

# inspired by Speak Now (Taylor’s Version)
# Lyrics from: Speak Now, Starlight, Sparks Fly, Haunted
sentences = [
"I am not the kind of girl, who should be rudely barging in on a white veil occasion, but you are not the kind of boy who should be marrying the wrong girl",
"I sneak in and see your friends and her snotty little family all dressed in pastel and she is yelling at a bridesmaid somewhere back inside a room wearing a gown shaped like a pastry",
"This is surely not what you thought it would be.",
"I lose myself in a daydream where I stand and say, ‘Don’t say yes, run away now I’ll meet you when you’re out of the church at the back door.'",
"Don’t wait, or say a single vow you need to hear me out and they said, ‘Speak now’.",
"Fond gestures are exchanged.",
"And the organ starts to play a song that sounds like a death march.",
"And I am hiding in the curtains, it seems that I was uninvited by your lovely bride-to-be.",
"She floats down the aisle like a pageant queen.",
"But I know you wish it was me you wish it was me don’t you?",
"I hear the preacher say, ‘Speak now or forever hold your peace'",
"There’s the silence, there’s my last chance.",
"I stand up with shaky hands, all eyes on me",
"Horrified looks from everyone in the room but I’m only looking at you.",
"And you’ll say, ‘Let’s run away now’ I’ll meet you when I’m out of my tux at the back door",
"Baby, I didn’t say my vows So glad you were around When they said, ‘Speak now'",
"I said, ‘Oh my, what a marvelous tune'",
"It was the best night, never would forget how we moved.",
"The whole place was dressed to the nines and we were dancing, dancing like we’re made of starlight",

"I met Bobby on the boardwalk summer of ’45",
"Picked me up late one night out the window we were seventeen and crazy running wild, wild.",
"Can’t remember what song he was playing when we walked in.",
"The night we snuck into a yacht club party pretending to be a duchess and a prince.",
"He said, ‘Look at you, worrying so much about things you can’t change You’ll spend your whole life singing the blues If you keep thinking that way'",
"He was tryna to skip rocks on the ocean saying to me ‘Don’t you see the starlight, starlight don’t you dream impossible things'",
"Ooh, ooh he’s talking crazy Ooh, ooh dancing with me Ooh, ooh we could get married Have ten kids and teach ’em how to dream",
"The way you move is like a full on rainstorm.",
"And I’m a house of cards",
"You’re the kind of reckless that should send me running but I kinda know that I won’t get far",
"And you stood there in front of me just close enough to touch",
"Close enough to hope you couldn’t see what I was thinking of",
"Drop everything now",
"Meet me in the pouring rain",
"Kiss me on the sidewalk",
"Take away the pain",
"Cause I see sparks fly, whenever you smile",
"Get me with those green eyes, baby as the lights go down",
"Gimme something that’ll haunt me when you’re not around",
"My mind forgets to remind me you’re a bad idea",
"You touch me once and it’s really something you find I’m even better than you imagined I would be",
"I’m on my guard for the rest of the world but with you, I know it’s no good"
"And I could wait patiently but I really wish you would"
"I run my fingers through your hair and watch the lights go wild",
"Just keep on keeping your eyes on me it’s just wrong enough to make it feel right",
"And lead me up the staircase won’t you whisper soft and slow, I’m captivated by you, baby like a fireworks show",
"You and I walk a fragile line I have known it all this time, But I never thought I’d live to see it break",
"It’s getting dark and it’s all too quiet And I can’t trust anything now And it’s coming over you like it’s all a big mistake",
"Oh, I’m holding my breath Won’t lose you again",
"Something’s made your eyes go cold",
"Come on, come on, don’t leave me like this I thought I had you figured out",
"Something’s gone terribly wrong you’re all I wanted",
"Can’t breathe whenever you’re gone can’t turn back now, I’m haunted",
"I just know You’re not gone, you can’t be gone, no",
]

DIMENSION=2000
# object should be inserted in the format of (title, date, location, speech embedding)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
]

COLLECTION_V12='COLLECTION_V12'
COLLECTION_V12_Q='COLLECTION_V12_Q'
schema = CollectionSchema(fields=fields)
collection_v12 = Collection(name=COLLECTION_V12, schema=schema)
collection_v12_ft5 = Collection(name=COLLECTION_V12_Q, schema=schema)

v12_embeds = {}
v12_q_embeds = {}
for sentence in sentences:
    v12_embeds[sentence] = v12.encode(sentence)
    print(v12_embeds[sentence].size())
    v12_q_embeds[sentence] = ft5_v12.encode(sentence) 

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 4},
}
collection_v12.create_index(field_name="embedding", index_params=index_params)
collection_v12.load()
collection_v12_ft5.create_index(field_name="embedding", index_params=index_params)
collection_v12_ft5.load()

for sentence in sentences:
    v12_insert = [
        {
            "sentence": sentence,
            "embedding": v12_embeds[sentence]
        }
    ]
    ft_insert = [
        {
            "sentence": sentence,
            "embedding": v12_q_embeds[sentence]
        }
    ]

 
    collection_v12.insert(v12_insert)
    collection_v12_ft5.insert(ft_insert)
 
 
collection_v12.flush()
collection_v12_ft5.flush()


search_embeds = {}
search_data = []
for sentence in sentences[0:2]:
    vector_embedding = ft3_v12.encode(sentence)
    search_embeds[sentence] = vector_embedding
    search_data.append(vector_embedding)


# now search
    
start1 = time()
res_v12 = collection_v12.search(
    data=search_data, # Embeded search value
    anns_field="embedding", # Search across embeddings
    param={
        "metric_type": "L2",
        "params": {"nprobe": 2}
    },
    limit = 3, # Limit to top_k results per search
    output_fields=["sentence"])
time1 = time() - start1
print(f"Time for first search: {time1}")
start2 = time()
res_v12_ft5 = collection_v12_ft5.search(
    data=search_data, # Embeded search value
    anns_field="embedding", # Search across embeddings
    param={"metric_type": "L2",
        "params": {"nprobe": 2}},
    limit = 3, # Limit to top_k results per search
    output_fields=["sentence"])
time2 = time() - start2
print(f"Time for second search: {time2}")