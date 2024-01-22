#! /usr/bin/env python3
from simile.tb_readers import *
from simile.milvus_store import *
from simile.vecstore import *
from itertools import islice

def p(args): print(args)

class Main:    
    def main(self, colName):
        enc = EncoderFactory.all_MiniLM_L6_v2()
        milv = MilvusVecstore(collectionName=colName, storeContent=True)
        # p(milv.listCollections())
        milv.collection.compact()
        p(milv.collection.primary_field)
        p(milv.collection.is_empty)
        p(milv.collection.num_entities)
        
        tbdl = TextbaseDownloads()
        p(f'will import from {tbdl.basedir}')
        sentences = tbdl.sentences()
        # sentences = islice(sentences, 200)
        i = 0
        for s in sentences:
            # p (f"{i} : encoding {s.path} / {s.location} : [{s.content}]")
            emb = enc.encode(s.content, show_progress_bar=False)
            id = s.id(tbdl.basedir)
            p (f'#{i} : {id} : len={len(s.content)}')
            # p(f'milv.put({id}, {len(emb)})')
            milv.put(id, emb, s.content)
            if (i % 5000 ==  0):
                p("flushing")
                milv.flush()
            i += 1
            
        p('done, flushing...')
        milv.flush()
        p('done, will create index...')
        milv.createIndexOnEmbeddings()
        milv.collection.load()
        
        resp = milv.collection.query(expr='id != ""', output_fields=['id'])
        # p(resp)
        p(len(resp))
        p(milv.count())

if __name__ == '__main__':
    import argparse
    import os
    
    defaultCollectionName = "textbase_dl"
 
    parser = argparse.ArgumentParser(description='Print the sum of two numbers')
    
    parser.add_argument('-d', type=str, help='Textbase downloads directory', required=True)
    parser.add_argument('-c', type=str, help='Milvus collection name', default=defaultCollectionName)

    args = parser.parse_args()

    tbdir = os.path.expanduser(args.d)
    colName = args.c
    
    if not os.path.isdir(tbdir):
        raise(Exception("not a directory: %s" % tbdir))

    Main().main(colName=colName)
