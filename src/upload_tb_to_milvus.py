#! /usr/bin/env python3
from simile.tb_readers import *
from simile.milvus_store import *
from simile.vecstore import *
from simile.encoder import *
from itertools import islice
from simile.util import p, StopWatch
from simile.minio_is_a_map import *


import enum
class ImportType(enum.Enum):
    SENTENCE=0
    PARAGRAPH=1

# this saves the last import in a local file, can be use for --resume function
class LastImportBookmark:
    
    filelocation = os.path.expanduser('~/.milvus-uploader-last-import')
    
    def setLastImport(self, lastImport: str):
        f = open(self.filelocation, "w")
        f.write(lastImport)
        f.close()
        
    def getLastImport(self) -> str:
        f = open(self.filelocation, "r")
        if not self.exists():
            return None
        resp = f.read()
        f.close()
        return resp
        
    def exists(self) -> bool:
        return os.path.exists(self.filelocation)
        
    def delete(self):
        if self.exists():
            os.remove(self.filelocation)
        

class Uploader:
    '''
        bulk uploads data into milvus.
        it does not care about already existing data. simply take all incoming 
        vectors, generates bulk files to import into milvus.
        no complicated logic to be kept into Python.
    '''
    encoder: Encoder
    collectionName: str
    milvusVectore: MilvusVecstore
    batchSize: int # how many entities will be sent at once to encoding and to milvus.
    importType: ImportType
    # milvusDataDir = os.path.expanduser('~/docker-volumes-nobkp/milvus/milvus/')
    bucket: MinioBucket
    
    def __init__(self, textbaseDownloads: TextbaseDownloads, 
                 collectionName: str, 
                 minio: MinioServer,
                 address: str = 'localhost:19530',
                 encoder = EncoderFactory.all_MiniLM_L6_v2(),
                 limit = -1,
                 batchSize = 2000,
                #  forceReimport = True,
                 importType: ImportType = ImportType.SENTENCE
                 ) -> None:
        self.bucket = minio['a-bucket']
        self.encoder = encoder
        self.collectionName = collectionName
        self.textbaseDownloads = textbaseDownloads
        self.limit = limit
        self.milvusServer=address
        self.batchSize = batchSize
        # self.forceReimport = forceReimport
        self.importType = importType
        # p(f'\t forceReimport = {self.forceReimport}')
    
    def upload(self):
        
        stopwatch =  StopWatch().start()
        
        enc = self.encoder
        colName = self.collectionName
        
        self.milvusVectore = MilvusVecstore(address=self.milvusServer, collectionName=colName)
        milv = self.milvusVectore
        # milv.collection.compact()
                
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
        # counterSkipped = 0
        while(True):
            crtBuffer = list(islice(ssentences, self.batchSize))
            if len(crtBuffer) == 0: break # this is really a do while
            
            ids = [it.getId() for it in crtBuffer]
            
            # # see what data can be skipped if the collection is not new and
            # # the forceReimport flag is set
            # if self.milvusVectore.collectionAlreadyExists and not self.forceReimport:
            #     # see if some of the data is not already in there
            #     existingIds = [it['id'] for it in self.milvusVectore.idsExist(ids)]
            #     notexistingIds = [it for it in ids if not it in existingIds]
            #     counterSkipped += len(notexistingIds)
            #     for id in existingIds:
            #         p (f'- SKIP #{i} : {id}')
            #         i += 1
                    
            #     # keep only non-existing (new)
            #     crtBuffer = [it for it in crtBuffer if it.getId() in notexistingIds]
            # # fi
            
            if len(crtBuffer) > 0:
                texts   = [it.text()  for it in crtBuffer]
                ids     = [it.getId() for it in crtBuffer] # again because crtBuffer may have changed
                
                p(f'sending {len(ids)} text chunks to encoder...')
                embeddingWatch = StopWatch().start()
                embeddings = enc.encode(sentences=texts, show_progress_bar=False)
                assert len(embeddings) == len(texts)
                
                p (f'batch #{batchCounter}. encoding took {embeddingWatch.stop()}')
                
                for id in ids:
                    p (f'> #{i} : {id}')
                    i += 1
                
                milvInsertionwatch = StopWatch().start()
                
                # milv.putAll(ids, embeddings)
                # uploadDir = os.path.join(self.milvusDataDir, enc.name)
                # uploadDir = self.milvusDataDir
                taskId = milv.bulkUploadLocalFS_rowbasedJson(
                    self.bucket,
                    batchCounter, 
                    ids, 
                    embeddings)
                yield taskId
                
                counterUploaded += len(embeddings)
                
                milv.flush()
                
                p(f"flushed. milvus insertion took {milvInsertionwatch.stop()}")
            
            batchCounter += 1
        # while
        
        milv.flush()
        milv.collection.compact()
        # milv.createIndex()
        
        p(f'.👑 upload finished. total entities: {milv.count()}. took {stopwatch.stop()}. bye.')
        return counterUploaded

if __name__ == '__main__':
    import argparse
    import os
    
    # defaultCollectionName = "textbase_sentences"
    defaultImportType = 'sentence'
 
    parser = argparse.ArgumentParser(description='Upload a directory of tb downloads to Milvus')
    
    parser.add_argument('-d', type=str, help='Textbase downloads directory', required=True)
    parser.add_argument('-a', type=str, help='Milvus server address', default='localhost:19530')
    parser.add_argument('-c', type=str, help='Milvus collection name', required=True)
    parser.add_argument('-t', type=str, help='Import type: sentence | paragraph', default='sentence')
    parser.add_argument('-f', type=bool, help='Force reimport', default=False)
    # parser.add_argument('--store_content', type=bool, help='Store content together with the vectors', default=False)

    args = parser.parse_args()

    tbdir = os.path.expanduser(args.d)
    colName                 = args.c
    importTypeArg: str      = args.t
    forceReimport           = args.f
    address                 = args.a

    importType: ImportType
    match importTypeArg.lower():
        case 'sentence' | 'sent' | 's':
            importType = ImportType.SENTENCE
        case 'paragraph' | 'para' | 'p':
            importType = ImportType.PARAGRAPH
        case _:
            raise Exception(f'unknown import type {importTypeArg}, sentence or paragraph')

    if not os.path.isdir(tbdir):
        raise(Exception("not a directory: %s" % tbdir))

    tbdl  = TextbaseDownloads(tbdir)
    uploader = Uploader(tbdl, colName, address=address, forceReimport=forceReimport, importType=importType)
    uploader.upload()
