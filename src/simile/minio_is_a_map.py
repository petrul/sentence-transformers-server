from typing import Any
from minio import Minio
from .util import *
import io
from typing import BinaryIO, ByteString

class MinioServer:

    minio: Minio
    
    def __init__(self, address = 'localhost:19533', 
                 username: str = "minioadmin", 
                 password: str = "minioadmin") -> None:
        self.minio = Minio(address,
                           access_key=username,
                           secret_key=password,
                           secure=False)
        
    def __getitem__(self, key: str):
        return MinioBucket(self, key)

class ContentAndMetadata:
    CONTENT_TYPE = 'contentType'
    LENGTH = 'length'
    
    metadata: dict = {
        CONTENT_TYPE: 'application/octet-stream',
        LENGTH: -1
    }
    content: BinaryIO
    
    @staticmethod
    def fromStr(s: str):
        resp = ContentAndMetadata()
        resp.setLength(len(s))
        resp.content = io.BytesIO(s.encode('utf-8'))
        resp.setContentType('text/plain')
        return resp
    
    @staticmethod
    def fromJson(s: str):
        resp = ContentAndMetadata()
        resp.setLength(len(s))
        resp.content = io.BytesIO(s.encode('utf-8'))
        resp.setContentType('application/json')
        return resp
    
    def getContentType(self):
        return self.metadata['contentType']

    def setContentType(self, value: str):
        self.metadata[self.CONTENT_TYPE] = value
        
    def setLength(self, length: int):
        self.metadata[self.LENGTH] = length
        
    def __len__(self):
        return self.metadata[self.LENGTH]

class MinioBucket:
    minio: MinioServer
    bucketName: str
    
    def __init__(self, minio: MinioServer, bucketName: str) -> None:
        self.minio = minio
        self.bucketName = bucketName
    
    def __getitem__(self, key: str): 
        return self.minio.minio.get_object(self.bucketName, key)
    
    def __setitem__(self, key: str, value: str | ContentAndMetadata):
        
        length = len(value)
        
        cm: ContentAndMetadata
        if type(value) == ContentAndMetadata:
            cm = value
        elif type(value) == str:    
            cm = ContentAndMetadata.fromStr(value)
        else:
            cm  = ContentAndMetadata(value)
            
        return self.minio.minio.put_object(self.bucketName, 
                                           key, 
                                           cm.content,
                                           length=length, 
                                           content_type=cm.getContentType())

    def delete(self, key: str):
        self.minio.minio.remove_object(self.bucketName, key)
        
if __name__ == '__main__':
    m = MinioServer()
    b = m['a-bucket']
    p(b)
    try:
        p(b['hello'])
        raise Exception('should fail')
    except:
        pass
    b['hello'] = 'world'
    p(b['hello'])
    b.delete('hello')