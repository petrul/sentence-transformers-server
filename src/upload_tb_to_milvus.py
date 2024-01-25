#! /usr/bin/env python3
from simile.tb_readers import *
from simile.milvus_store import *
from simile.vecstore import *
from simile.encoder import *
from itertools import islice
from simile.util import p
import time

class StopWatch:
    
    def start(self):
        self.timestamp_start = time.time()
        return self
        
    def stop(self):
        self.timestamp_end = time.time()
        self.took = self.timestamp_end - self.timestamp_start
        return self.took
    
    def __str__(self) -> str:
        return str(".2f" % self.took)

class Uploader:
    
    encoder: Encoder
    collectionName: str
    milvusVectore: MilvusVecstore
    batchSize: int
    
    def __init__(self, textbaseDownloads: TextbaseDownloads, 
                 collectionName: str, 
                 encoder = EncoderFactory.all_MiniLM_L6_v2(),
                 limit = -1,
                 batchSize = 1000,
                 forceReimport = True) -> None:
        self.encoder = encoder
        self.collectionName = collectionName
        self.textbaseDownloads = textbaseDownloads
        self.limit = limit
        self.batchSize = batchSize
        self.forceReimport = forceReimport
        p(f'\t forceReimport = {self.forceReimport}')
    
    def upload(self):
        
        stopwatch =  StopWatch().start()
        
        enc = self.encoder
        colName = self.collectionName
        
        self.milvusVectore = MilvusVecstore(collectionName=colName)
        milv = self.milvusVectore
        milv.collection.compact()
                
        tbdl = self.textbaseDownloads
        p(f'will import from {tbdl.basedir}')
        ssentences = tbdl.significant_sentences()
        if self.limit > 0 : ssentences = islice(ssentences, self.limit)
        
        batchCounter = 0
        i = 0 # general counter, skipped and unskipped
        counterUploaded = 0
        counterSkipped = 0
        while(True):
            crtBuffer = list(islice(ssentences, self.batchSize))
            if len(crtBuffer) == 0: break # this is really a do while
            
            ids = [it.getId() for it in crtBuffer]
            
            # see what data can be skipped if the collection is not new and
            # the forceReimport flag is set
            if self.milvusVectore.collectionAlreadyExisted and not self.forceReimport:
                # see if some of the data is not already in there
                existingIds = [it['id'] for it in self.milvusVectore.idsExist(ids)]
                notexistingIds = [it for it in ids if not it in existingIds]
                counterSkipped += len(notexistingIds)
                for id in existingIds:
                    p (f'- SKIP #{i} : {id}')
                    i += 1
                    
                # keep only non-existing (new)
                crtBuffer = [it for it in crtBuffer if it.getId() in notexistingIds]
            #
            
            if len(crtBuffer) > 0:
                texts = [it.text() for it in crtBuffer]
                ids = [it.getId() for it in crtBuffer] # again because crtBuffer may have changed
                
                embeddingWatch = StopWatch().start()
                embeddings = enc.encode(sentences=texts, show_progress_bar=False)
                assert len(embeddings) == len(texts)
                
                p (f'batch #{batchCounter}. encoding took {embeddingWatch.stop()}')
                
                for id in ids:
                    p (f'> #{i} : {id}')
                    i += 1
                
                milvInsertionwatch = StopWatch().start()
                
                milv.putAll(ids, embeddings)
                counterUploaded += len(embeddings)
                
                milv.flush()
                
                p(f"flushed. milvus insertion took {milvInsertionwatch.stop()}")
            
            batchCounter += 1
        # done
        
        milv.flush()
        milv.collection.compact()
        milv.createIndex()
        
        p(f'.👑 upload finished. total entities: {milv.count()}. took {stopwatch.stop()}. bye.')
        return counterUploaded

if __name__ == '__main__':
    import argparse
    import os
    
    defaultCollectionName = "textbase_sentences"
 
    parser = argparse.ArgumentParser(description='Upload a directory of tb downloads to Milvus')
    
    parser.add_argument('--dl', type=str, help='Textbase downloads directory', required=True)
    parser.add_argument('--col', type=str, help='Milvus collection name', default=defaultCollectionName)
    # parser.add_argument('--store_content', type=bool, help='Store content together with the vectors', default=False)

    args = parser.parse_args()

    tbdir = os.path.expanduser(args.dl)
    colName = args.col
    # store_content: bool = args.store_content
    
    if not os.path.isdir(tbdir):
        raise(Exception("not a directory: %s" % tbdir))

    tbdl  = TextbaseDownloads(tbdir)
    uploader = Uploader(colName)
    uploader.upload()
