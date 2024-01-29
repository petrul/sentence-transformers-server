from .vecstore import Store
import shutil
import os

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    BulkInsertState
)
import numpy
from .tb_readers import *

class MilvusVecstore(Store):

    fmt = "\n=== {:30} ===\n"    
    collection: Collection
    maxContentLength=5000 # varchar length for field content
    collectionAlreadyExists: bool  # true if collection was already there
    
    # vector dim is the size of the vectors that you expect to store
    def __init__(self, 
                 address = "localhost:19530",
                 collectionName="default",
                 vector_dimension=384):
        p(f'MilvusVecstore @{address}')
        connections.connect("default",  address=address)
        self.createCollection(collectionName, vector_dimension)
        
    def createCollection(self, collectionName, vector_dimension):
        self.collectionAlreadyExists = utility.has_collection(collectionName)
        p(f"Does collection [{collectionName}] exist in Milvus: {self.collectionAlreadyExists}")
                
        if self.collectionAlreadyExists:
            self.collection = Collection(collectionName)
            p(f'loading collection {collectionName} with {self.collection.num_entities} existing entities')
            self.collection.load()

        else:
            fields: list
            # if (self.storeContent):
            #     fields = [
            #         FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=500),
            #         FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=self.maxContentLength),
            #         FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=vector_dimension)
            #     ]
            # else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=500),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=vector_dimension)
            ]

            schema = CollectionSchema(fields, "Embedding vectors representing text")

            p(f"created collection [{collectionName}]")
            self.collection = Collection(collectionName, schema, consistency_level="Strong")


    def createIndex(self):
        self.createIndex_DISKANN_OnEmbeddings()
        # self.createIndex_IVFFLAT_OnEmbeddings()

    # NB: this might eat up RAM
    def createIndex_IVFFLAT_OnEmbeddings(self):
        print(self.fmt.format("Start Creating index IVF_FLAT"))
        index = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            }

        self.collection.create_index("embeddings", index)

    def createIndex_DISKANN_OnEmbeddings(self):
        print(self.fmt.format("Start Creating index DISKANN"))
        index_params = {
            "index_type": "DISKANN",
            "metric_type": "L2",
            "params": {}
        }

        self.collection.create_index("embeddings", index_params)

    def put(self, key: object, value: numpy.ndarray):
        self.putAll([key], [value])    
        
    def putAll(self, keys: list[object], values: list[numpy.ndarray]):
        data = [
            keys,
            values
        ]
        insert_result = self.collection.insert(data)
        return insert_result

    def bulkUploadFiles(self, uploadDir: str, batchCounter: int, keys: list[object], values: list[numpy.ndarray]):
        if not os.path.exists(uploadDir): os.makedirs(uploadDir)
        # import numpy
        # numpy.save('batch.npy', numpy.array([101, 102, 103, 104, 105]))
        # numpy.save('word_count.npy', numpy.array([13, 25, 7, 12, 34]))
        # arr = numpy.array([[1.1, 1.2],
        #     [2.1, 2.2],
        #     [3.1, 3.2],
        #     [4.1, 4.2],
        #     [5.1, 5.2]])
        # numpy.save('book_intro.npy', arr)
        
        rows = []
        for id, vec in zip(keys, values):
            rows.append({
                'id': id,
                'embeddings': vec.tolist()
            })

        import json

        # Data to be written
        data = {
            "rows": rows
        }
        
        dumpName = f'{self.collection.name}-#{batchCounter}.json'
        with open(dumpName, "w" ) as fh:
            json.dump( data , fh )
            
        # from pymilvus import utility
        task_id = utility.do_bulk_insert(
            collection_name=self.collection.name,
            is_row_based=True,
            files=[ dumpName ]
        )
        self.print_task(task_id)
        
    def print_task(self, id: int):
        task = utility.get_bulk_insert_state(task_id=id)
        print("Task state:", task.state_name)
        print("Imported files:", task.files)
        print("Collection name:", task.collection_name)
        print("Partition name:", task.partition_name)
        print("Start time:", task.create_time_str)
        print("Imported row count:", task.row_count)
        print("Entities ID array generated by this task:", task.ids)

        if task.state == BulkInsertState.ImportFailed:
            print("Failed reason:", task.failed_reason)
            

    def flush(self):
        self.collection.flush()

    def listIds(self):
        # self.collection.search
        pass
    
    # returns, amongst the given list of ids, those that actually exist in the collection
    def idsExist(self, ids: list[str]):
        self.collection.load()
        ids_str = str(ids)
        return self.collection.query(expr=f'id in {ids_str}')

    def count(self):
        return self.collection.num_entities

    def listCollections(self):
        return utility.list_collections()

def p(args): print(args)
        
if __name__ == '__main__':
    pass    
