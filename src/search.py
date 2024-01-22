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
            "params": {"nprobe": 10}
        }
        resp = milv.collection.search([emb], 'embeddings', search_params, 10)
        
        # p(resp)
        p(type(resp))
        p(type(resp[0]))
        hits = resp[0]
        p(hits[0])
        p(type(hits))
        
        hhits = [str(it) for it in hits]
        p(hhits)
        # p('\n'.join(hhits))
        p('================')
        p([ it.id for it in hits])
        
        dlsentences = [ DlSentence.fromMilvusId(tbdl.basedir, it.id) for it in hits]
        p('\n'.join([it.text() for it in dlsentences]))
        
        # p('\n'.join(hits))
        
        # p('\n'.join(resp))
        

        # p(f'will import from {tbdl.basedir}')
        # sentences = tbdl.sentences()
        # sentences = islice(sentences, 200)
        i = 0
        # for s in sentences:
        #     # p (f"{i} : encoding {s.path} / {s.location} : [{s.content}]")
            
        #     id = s.id(tbdl.basedir)
        #     if store_content:
        #         p (f'#{i} : {id} : len={len(s.content)}')
        #     else:
        #         p (f'#{i} : {id}')
                
        #     milv.put(id, emb, s.content)
        #     if (i % 5000 ==  0):
        #         p("flushing")
        #         milv.flush()
        #     i += 1
            
        # p('done, flushing...')
        # milv.flush()
        # p('done, will create index...')
        # milv.createIndex_IVFFLAT_OnEmbeddings()
        # milv.collection.load()
        
        # resp = milv.collection.query(expr='id != ""', output_fields=['id'])
        # # p(resp)
        # p(len(resp))
        # p(milv.count())

if __name__ == '__main__':
    import argparse
    import os
    
    defaultCollectionName = "textbase_dl"
 
    parser = argparse.ArgumentParser(description='Search')
    
    Searcher(defaultCollectionName).search("/home/petru/data/textbase-dl", 
                                           "Where was Jesus born?")
    
    quit()
    parser.add_argument('--dl', type=str, help='Textbase downloads directory', required=True)
    parser.add_argument('-address', type=str, help='Milvus server address', default="mini.local:19530")
    parser.add_argument('-col', type=str, help='Milvus collection name', default=defaultCollectionName)

    args, unknown  = parser.parse_known_args()
    
    query_str =  ' '.join(unknown)
    colName = args.col
    basedir = os.path.expanduser(args.dl)
    
    
    Searcher(colName).search(basedir, query_str)

    # tbdir = os.path.expanduser(args.dl)
    
    # store_content: bool = args.store_content
    
    # if not os.path.isdir(tbdir):
    #     raise(Exception("not a directory: %s" % tbdir))

    # Main().main(colName=colName, store_content=store_content)
