#! /usr/bin/env python3
from simile.tb_readers import *
from simile.milvus_store import *
from simile.vecstore import *
from simile.encoder import *
from itertools import islice

def p(args): print(args)

class Uploader:
    
    encoder: Encoder
    collectionName: str
    milvusVectore: MilvusVecstore
    
    
    def __init__(self, textbaseDownloads: TextbaseDownloads, 
                 collectionName: str, 
                 encoder = EncoderFactory.all_MiniLM_L6_v2(),
                 limit = -1) -> None:
        self.encoder = encoder
        self.collectionName = collectionName
        self.textbaseDownloads = textbaseDownloads
        self.limit = limit
    
    def upload(self, store_content: bool):
        enc = self.encoder
        colName = self.collectionName
        
        self.milvusVectore = MilvusVecstore(collectionName=colName, storeContent=store_content)
        milv = self.milvusVectore
        milv.collection.compact()
                
        tbdl = self.textbaseDownloads
        p(f'will import from {tbdl.basedir}')
        ssentences = tbdl.significant_sentences()
        if self.limit > 0 : ssentences = islice(ssentences, self.limit)
        
        # i = 0
        for i, s in enumerate(ssentences):
            stext = s.text()
            emb = enc.encode(stext, show_progress_bar=False)
            id = s.getId()
            if store_content:
                p (f'#{i} : {id} : len={len(stext)}')
            else:
                p (f'#{i} : {id}')
                
            milv.put(id, emb, stext)
            if (i % 5000 ==  0):
                p("flushing")
                milv.flush()
            # i += 1
            
        p('done, flushing...')
        milv.flush()
        p('done, will create index...')
        milv.collection.compact()
        milv.createIndex()
        
        p(f'total entities: {milv.count()}')

if __name__ == '__main__':
    import argparse
    import os
    
    defaultCollectionName = "textbase_sentences"
 
    parser = argparse.ArgumentParser(description='Upload a directory of tb downloads to Milvus')
    
    parser.add_argument('--dl', type=str, help='Textbase downloads directory', required=True)
    parser.add_argument('--col', type=str, help='Milvus collection name', default=defaultCollectionName)
    parser.add_argument('--store_content', type=bool, help='Store content together with the vectors', default=False)

    args = parser.parse_args()

    tbdir = os.path.expanduser(args.dl)
    colName = args.col
    store_content: bool = args.store_content
    
    if not os.path.isdir(tbdir):
        raise(Exception("not a directory: %s" % tbdir))

    tbdl  = TextbaseDownloads(tbdir)
    uploader = Uploader(colName)
    uploader.upload(store_content=store_content)
