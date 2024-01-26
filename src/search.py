#! /usr/bin/env python3
from simile.tb_readers import *
from simile.milvus_store import *
from simile.vecstore import *
from itertools import islice
from simile.encoder import EncoderFactory

def p(*args): print(*args)

class Searcher:
    
    milvusAdress: str
    
    def __init__(self, colectionName, milvusAddress: str = 'localhost:19530') -> None:
        self.colectionName  = colectionName
        self.milvusAddress = milvusAddress
      
    def search(self,  
               dlbasedir, 
               query: str, 
               max_results: int = 10,
               ):
        enc = EncoderFactory.all_MiniLM_L6_v2()
        milv = MilvusVecstore(collectionName=self.colectionName, address=self.milvusAddress)
        tbdl = TextbaseDownloads(dlbasedir)
        
        assert not milv.collection.is_empty
        emb = enc.st.encode(query, show_progress_bar=False)
        milv.collection.load()
        search_params = {
            "metric_type": "L2", 
            # "metric_type": "IP", 
            "offset": 0, 
            "ignore_growing": False, 
            "params": {"nprobe": 20}
        }
        resp = milv.collection.search([emb], 'embeddings', search_params, max_results)
        hits = resp[0]
        
        dlsentences = [(
                            DlSentence.fromMilvusId(tbdl.basedir, it.id),
                            it.distance
                        )
                       for it in hits ]
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
    parser.add_argument('--address', type=str, help='Milvus server address', default="mini.local:19530")
    parser.add_argument('--collection', type=str, help='Milvus collection name', default=defaultCollectionName)
    parser.add_argument('-n', type=int, help='max results nr', default=10)

    args, unknown  = parser.parse_known_args()
    query_str =  ' '.join(unknown)
    
    colName = args.collection
    basedir = os.path.expanduser(args.dl)
    max_results = args.n
    milvusAddress = args.address

    if query_str.strip() == '':
        raise Exception('please provide a text to search')
    
    p("******************")
    p("* MILVUS SEARCHER")
    p("******************")
    p(f'<( Query )> : \n\n[{query_str}]\n')
    p('=========')
    hits = Searcher(colName, milvusAddress=milvusAddress).search(basedir, query_str, max_results)
    
    for (i, (sent, dist))  in enumerate(hits):
        p(f'<( #{i + 1} )> {sent.paragraph.file.getTextbaseUrl()} - {dist} :\n')
        # p(f'{sent.getId()} - {dist} :')
        # p(f'{sent.paragraph.file.getCompletePath()} :')
        # p(f'{sent.paragraph.file.getTextbaseUrl()} - {dist} :')
        
        p(f'\t{sent.text()}\n')
