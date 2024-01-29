from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import numpy
import pickle
import os
from .util import *

class Cache:
    def put(self, key: str, value: numpy.ndarray):
        pass
    
    def has(self, key: str) -> bool:
        return False

    def get(self, key: str) -> numpy.ndarray:
        return None

    
# this might be used for caching encodings on-disk if upload proves to be difficult
# so that they be not re-computed every time.
class VectorCache(Cache):

    cachedir: str

    def __init__(self, cachedir: str):
        self.cachedir = cachedir
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)

    def __getCompleteFilename(self, filename: str):
        fragmName = self.__getFragmentedName(filename)
        resp = os.path.join(self.cachedir, fragmName)
        return resp


    def __getFragmentedName(self, filename: str):
        assert len(filename) > 3 * 2
        d1 = filename[0:2]
        d2 = filename[2:4]
        d3 = filename[4:6]
        return f'{d1}/{d2}/{d3}/{filename}'

    def put(self, key: str, value: numpy.ndarray):
        path = self.__getCompleteFilename(key)
        parentDir = os.path.dirname(path)
        if not os.path.exists(parentDir):
            os.makedirs(parentDir)
        fh = open(path, 'wb')
        pickle.dump(value, fh)
        fh.close()

    def has(self, key: str) -> bool:
        filename = self.__getCompleteFilename(key);
        return os.path.exists(filename)

    def get(self, key: str) -> numpy.ndarray:
        filename = self.__getCompleteFilename(key)
        fh = open(filename, 'rb')
        resp = pickle.load(fh)
        fh.close()
        return resp
    
    def rm_rf_cachedir(self, iUnderstandThatThisIsAPotentiallyDestructiveOperation: bool = False):
        if not iUnderstandThatThisIsAPotentiallyDestructiveOperation: raise Exception()
        import shutil
        shutil.rmtree(self.cachedir)


class Encoder:
    name: str
    st: SentenceTransformer
    cache: Cache
    
    def __init__(self, name, st: SentenceTransformer, cache: Cache = Cache()):
        self.name = name
        self.st = st
        self.cache = cache
    
    def __sha1(self, text):
        import hashlib
        sha1encoder = hashlib.sha1()
        sha1encoder.update(text.encode('utf-8'))
        return sha1encoder.hexdigest()            
    
    def encode(self, sentences: list[str], **args) -> list[numpy.ndarray]:
        j2i = {}
        uncached_sentences = []
        resp = list(range(len(sentences)))
                
        sha1s = [ self.__sha1(it) for it in sentences ]
        
        j = 0
        for i, s in enumerate(sentences):
            sha1 = sha1s[i]
            if self.cache.has(sha1):
                resp[i] = self.cache.get(sha1)
            else:
                j2i[j] = i
                uncached_sentences.append(s)
                j += 1
        
        assert len(j2i) == len(uncached_sentences)
        computed = self.st.encode(sentences=uncached_sentences, **args) 
        for j, vect in enumerate(computed):
            i = j2i[j]
            resp[i] = vect
            sha1 = sha1s[i]
            # assert not self.cache.has(sha1) 
            self.cache.put(sha1, vect)
        
        assert len(resp) == len(sentences)
        return resp

class EncoderFactory: 

    @staticmethod
    def all_MiniLM_L6_v2(cache = Cache()):
        name = 'all-MiniLM-L6-v2'
        st = SentenceTransformer(name)
        p('model [%s], with max_seq_length %d' % (name, st.get_max_seq_length()))        
        return Encoder(name, st, cache=cache)

