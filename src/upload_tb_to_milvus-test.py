from simile.tb_readers import *
from simile.util import *
from itertools import islice
from upload_tb_to_milvus import *
import unittest
import os

class UploaderTest(unittest.TestCase):
    textbase_downloads_dir = os.path.abspath(f'{scriptDir()}/../../tests/resources/textbase-dl')
    
            
    def testUploadSentences(self):
        tbdl = TextbaseDownloads(self.textbase_downloads_dir)
        
        colname = 'test_sentences_' + randomAlphabetic(10)
        
        limit = 20
        uploader = Uploader(tbdl, colname, limit=limit, batchSize=3, forceReimport=False)
        
        # 1. all insert
        uploader.upload()
        assert uploader.milvusVectore.count() == limit
        
        # 2. all skip
        uploader.upload() # again, all records should be skipped
        assert uploader.milvusVectore.count() == limit
        
        
        # 3. now first are already inserted while the latter half is new
        limit = 40
        uploader = Uploader(tbdl, colname, limit=limit, batchSize=10, forceReimport=False)
        uploader.upload() # again, all records should be skipped
        
        uploader.milvusVectore.collection.drop()
        
    def testUploadParagraphs(self):
        tbdl = TextbaseDownloads(self.textbase_downloads_dir)
        
        colname = 'test_paras_' + randomAlphabetic(10)
        
        limit = 5
        uploader = Uploader(tbdl, colname, limit=limit, 
                            batchSize=3, forceReimport=False, 
                            importType=ImportType.PARAGRAPH)
        
        # 1. all insert
        uploader.upload()
        assert uploader.milvusVectore.count() == limit
        
        uploader.milvusVectore.collection.drop()
       
        

if __name__ == '__main__':

    # UploaderTest().testUploadSentences()
    UploaderTest().testUploadParagraphs()