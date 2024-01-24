from simile.tb_readers import *
from simile.util import *
from itertools import islice
from upload_tb_to_milvus import *
import unittest
import os

class UploaderTest(unittest.TestCase):
    textbase_downloads_dir = os.path.abspath(f'{scriptDir()}/../../tests/resources/textbase-dl')
    
            
    def testUpload(self):
        tbdl = TextbaseDownloads(self.textbase_downloads_dir)
        colname = 'test_' + randomAlphabetic(10)
        p(colname)
        limit = 20
        uploader = Uploader(tbdl, colname, limit=limit)
        uploader.upload(False)
        assert uploader.milvusVectore.count() == limit
        uploader.milvusVectore.collection.drop()


if __name__ == '__main__':
    UploaderTest().testUpload()