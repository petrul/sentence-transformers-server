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

import enum
class ImportType(enum.Enum):
    SENTENCE=0
    PARAGRAPH=1

class Uploader:
    
    encoder: Encoder
    collectionName: str
    milvusVectore: MilvusVecstore
    batchSize: int # how many entities will be sent at once to encoding and to milvus.
    importType: ImportType
    
    def __init__(self, textbaseDownloads: TextbaseDownloads, 
                 collectionName: str, 
                 encoder = EncoderFactory.all_MiniLM_L6_v2(),
                 limit = -1,
                 batchSize = 2000,
                 forceReimport = True,
                 importType: ImportType = ImportType.SENTENCE
                 ) -> None:
        self.encoder = encoder
        self.collectionName = collectionName
        self.textbaseDownloads = textbaseDownloads
        self.limit = limit
        self.batchSize = batchSize
        self.forceReimport = forceReimport
        self.importType = importType
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
        
        ssentences = None
        match self.importType:
            case  ImportType.SENTENCE:
                ssentences = tbdl.significant_sentences()
            case ImportType.PARAGRAPH:
                ssentences = tbdl.paragraphs()

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
            if self.milvusVectore.collectionAlreadyExists and not self.forceReimport:
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
                
                p(f'sending {len(ids)} text chunks to encoder...')
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
    
    # defaultCollectionName = "textbase_sentences"
    defaultImportType = 'sentence'
 
    parser = argparse.ArgumentParser(description='Upload a directory of tb downloads to Milvus')
    
    parser.add_argument('-d', type=str, help='Textbase downloads directory', required=True)
    parser.add_argument('-c', type=str, help='Milvus collection name', required=True)
    parser.add_argument('-t', type=str, help='Import type: sentence | paragraph', default='sentence')
    parser.add_argument('-f', type=bool, help='Force reimport', default=False)
    # parser.add_argument('--store_content', type=bool, help='Store content together with the vectors', default=False)

    args = parser.parse_args()

    tbdir = os.path.expanduser(args.d)
    colName = args.c
    importTypeArg: str = args.t
    forceReimport = args.f
    
    importType: ImportType
    match importTypeArg.lower():
        case 'sentence' | 'sent' | 's':
            importType = ImportType.SENTENCE
        case 'paragraph' | 'para' | 'p':
            importType = ImportType.PARAGRAPH
        case _:
            raise Exception(f'unknown import type {importTypeArg}, sentence or paragraph')
            
    # p(importType)
    # quit()
    # store_content: bool = args.store_content
    
    if not os.path.isdir(tbdir):
        raise(Exception("not a directory: %s" % tbdir))

    tbdl  = TextbaseDownloads(tbdir)
    uploader = Uploader(tbdl, colName, forceReimport=forceReimport)
    uploader.upload()
