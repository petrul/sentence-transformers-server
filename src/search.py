#! /usr/bin/env python3
from simile.tb_readers import *
from simile.milvus_store import *
from simile.vecstore import *
from itertools import islice

def p(*args): print(*args)

class Searcher:
    
    def __init__(self, colectionName) -> None:
        self.colectionName  = colectionName
      
    def search(self,  dlbasedir, query: str):
        enc = EncoderFactory.all_MiniLM_L6_v2()
        milv = MilvusVecstore(collectionName=self.colectionName, storeContent=False)
        tbdl = TextbaseDownloads(dlbasedir)
        
        assert not milv.collection.is_empty
        emb = enc.encode(query, show_progress_bar=False)
        milv.collection.load()
        search_params = {
            "metric_type": "L2", 
            "offset": 0, 
            "ignore_growing": False, 
            "params": {"nprobe": 20}
        }
        resp = milv.collection.search([emb], 'embeddings', search_params, 20)
        hits = resp[0]
        
        dlsentences = [ DlSentence.fromMilvusId(tbdl.basedir, it.id) for it in hits ]
        return dlsentences


if __name__ == '__main__':
    import argparse
    import os
    
    defaultCollectionName = "textbase_dl"
 
    parser = argparse.ArgumentParser(description='Search')
    
    # Searcher(defaultCollectionName).search("/home/petru/data/textbase-dl", 
    #                                        "Where was Jesus born?")    
    # quit()
    
    parser.add_argument('--dl', type=str, help='Textbase downloads directory', default='~/data/textbase-dl')
    parser.add_argument('-address', type=str, help='Milvus server address', default="mini.local:19530")
    parser.add_argument('-col', type=str, help='Milvus collection name', default=defaultCollectionName)

    args, unknown  = parser.parse_known_args()
    
    query_str =  ' '.join(unknown)
    colName = args.col
    basedir = os.path.expanduser(args.dl)

    if query_str.strip() == '':
        raise Exception('please provide a text to search')
    
    sentences = Searcher(colName).search(basedir, query_str)
    
    for sent in sentences:
            p(f'{sent.getId()}:')
            p(f'\t{sent.text()}')
