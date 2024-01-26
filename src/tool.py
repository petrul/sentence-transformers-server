#! /usr/bin/env python3
from simile.tb_readers import *
from simile.milvus_store import *
from simile.vecstore import *
from itertools import islice
from simile.encoder import EncoderFactory
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

def p(*args): print(*args)

class MilvusTool:
    
    milvusAdress: str
    
    def __init__(self, milvusAddress: str = 'localhost:19530') -> None:
        # self.colectionName  = colectionName
        self.milvusAddress = milvusAddress
      
    def dropIndex(self, collectionName: str): 
        connections.connect("default",  address=self.milvusAddress)
        collection = Collection(collectionName)  
        collection.drop_index()


if __name__ == '__main__':
    import argparse
    import os
    
    # defaultCollectionName = "textbase_dl"
 
    parser = argparse.ArgumentParser(description='Milvus tool')

    # parser.add_argument('-d', type=str, help='Textbase downloads directory', default='~/data/textbase-dl')
    parser.add_argument('-a', type=str, help='Milvus server address', default="localhost:19530")
    parser.add_argument('-c', type=str, help='Milvus collection name', required=True)
    # parser.add_argument('-n', type=int, help='max results nr', default=10)

    args, unknown  = parser.parse_known_args()
    query_str =  ' '.join(unknown)
    
    colName = args.c
    # basedir = os.path.expanduser(args.dl)
    # max_results = args.n
    milvusAddress = args.a

    tool = MilvusTool(milvusAddress=milvusAddress)
    tool.dropIndex(colName)