from simile.tb_readers import *
from simile.util import *
from simile.minio_is_a_map import *
from upload_tb_to_milvus import *
from itertools import islice

import unittest
import os
import time

class UploaderTest(unittest.TestCase):
    textbase_downloads_dir = os.path.abspath(f'{scriptDir()}/../../tests/resources/textbase-dl')
    minio = MinioServer()
    
    def testUploadSentences(self):
        tbdl = TextbaseDownloads(self.textbase_downloads_dir)
        
        colname = 'test_sentences_' + randomAlphabetic(10)
        
        limit = 20
        uploader = Uploader(tbdl, colname, self.minio, limit=limit, batchSize=3)
        
        # 1. all bulk upload
        taskids = [it for it in uploader.upload()]
        
        # time.sleep(1)
        
        for tid in taskids:
            bulkstate = utility.get_bulk_insert_state(tid)
            p(bulkstate)
            # p(type(bulkstate.state))
            # assert bulkstate.state ==  BulkInsertState.ImportPersisted
                    
        # p(utility.list_bulk_insert_tasks(collection_name=colname))
        
        p(uploader.milvusVectore.count())
        # assert uploader.milvusVectore.count() == limit
        
        # 2. all bulk upload again
        # uploader.upload() # again, all records should be skipped
        
        # p(utility.list_bulk_insert_tasks(collection_name=colname))
        
        # assert uploader.milvusVectore.count() == limit
        
        
        # 3. now first are already inserted while the latter half is new
        # limit = 40
        # uploader = Uploader(tbdl, colname, limit=limit, batchSize=10, forceReimport=False)
        # uploader.upload() # again, all records should be skipped
        
        uploader.milvusVectore.collection.drop()
        
    def testUploadParagraphs(self):
        tbdl = TextbaseDownloads(self.textbase_downloads_dir)
        
        colname = 'test_paras_' + randomAlphabetic(10)
        
        limit = 5
        uploader = Uploader(tbdl, colname, 
                            self.minio, 
                            limit=limit, 
                            batchSize=3,
                            importType=ImportType.PARAGRAPH)
        
        # 1. all insert
        uploader.upload()
        p(uploader.milvusVectore.count())
        # assert uploader.milvusVectore.count() == limit
        
        uploader.milvusVectore.collection.drop()
       
    def testLastImportBookmark(self):
        libmk = LastImportBookmark()
        libmk.delete()
        assert not libmk.exists()
        text = randomAlphabetic(10)
        libmk.setLastImport(text)
        assert libmk.exists()
        assert libmk.getLastImport() == text
        assert libmk.getLastImport() == text # again
        text2 = randomAlphabetic(12)
        libmk.setLastImport(text2)
        assert not libmk.getLastImport() == text
        assert libmk.getLastImport() == text2
        
        libmk.delete()
        assert not libmk.exists()
        

if __name__ == '__main__':
    unittest.main()
    # UploaderTest().testUploadSentences()
    # UploaderTest().testUploadParagraphs()
    # UploaderTest().testLastImportBookmark()
    