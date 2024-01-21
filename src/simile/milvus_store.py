from vecstore import Store, EncoderFactory
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import numpy
from tb_readers import *
from itertools import islice


class MilvusVecstore(Store):

    fmt = "\n=== {:30} ===\n"    
    collection: Collection
    
    # vector dim is the size of the vectors that you expect to store
    def __init__(self, address = "localhost:19530", collectionName="default", vector_dimension=384):
        p(self.fmt.format('starting Milvus'))
        connections.connect("default",  address=address)
        collectionExists = utility.has_collection("hello_milvus")
        p(f"Does collection [{collectionName}] exist in Milvus: {collectionExists}")
                
        if not collectionExists:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=500),
                # FieldSchema(name="random", dtype=DataType.DOUBLE),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=vector_dimension)
            ]

            schema = CollectionSchema(fields, "Embedding vectors representing text")

            p(f"Create collection [{collectionName}]")
            self.collection = Collection(collectionName, schema, consistency_level="Strong")
            
            print(self.fmt.format("Start Creating index IVF_FLAT"))
            index = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            }

            self.collection.create_index("embeddings", index)

    def put(self, key: object, value: numpy.ndarray):
        toInsert = [
            [key],
            [value]
        ]
        insert_result = self.collection.insert(toInsert)
        p(f"insert result {insert_result}")

    def flush(self):
        self.collection.flush()

    def listIds(self):
        # self.collection.search
        pass

    def count(self):
        return self.collection.num_entities

    def listCollections(self):
        return utility.list_collections()

def p(args): print(args)
        
if __name__ == '__main__':
    milv = MilvusVecstore(collectionName='textbase')
    p(milv.listCollections())
    milv.collection.compact()
    p(milv.collection.primary_field)
    p(milv.collection.is_empty)
    p(milv.collection.num_entities)

    enc = EncoderFactory.all_MiniLM_L6_v2()
    tbdl = TextbaseDownloads()
    sentences = tbdl.sentences()
    # sentences = islice(sentences, 200)
    i = 0
    for s in sentences:
        p (f"{i} : encoding {s.path} / {s.location} : [{s.content}]")
        emb = enc.encode(s.content)
        p(f'milv.put({s.id()}, {len(emb)})')
        milv.put(s.id(), emb)
        if (i % 1000 ==  0): 
            p("flushing")
            milv.flush()
        i += 1
        
    milv.flush()
    milv.collection.load()
    
    resp = milv.collection.query(expr='id != ""', output_fields=['id'])
    # p(resp)
    p(len(resp))
    p(milv.count())
    
