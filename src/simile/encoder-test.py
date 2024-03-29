import unittest
from util import *
from encoder import *

class EncoderTest(unittest.TestCase):
    import tempfile
    import os
    
    tmp_dir = os.path.join(tempfile.gettempdir(), 'TextbaseReadersTest_' + randomAlphabetic(3))
    res_dir = os.path.abspath(f'{scriptDir()}/../../tests/resources/textbase-dl')
            
    def testCaching(self):
        
        cacheFact: CacheFactory = VectorCacheFactory(self.tmp_dir)
        enc = EncoderFactory(cacheFactory=cacheFact).all_MiniLM_L6_v2()
        
        sentences = [
            "foaie verde",
            "gigea",
            "ma cam doare capul",
            "ma cam doare capul",
        ]
        
        resp1 = enc.encode(sentences)
        
        import os
        assert os.path.exists(os.path.join(enc.cache.cachedir, '66/d3/17/66d317cc6ddc5cccc2cf0a350a33fcb2ac4c9842'))
        
        resp2 = enc.encode(sentences)
        
        assert len(resp1) == len(resp2)
        assert len(resp1) == len(sentences)
        
        for i, _ in enumerate(resp1):
            resp1[i] == resp2[i]
        
        enc.cache.rm_rf_cachedir(iUnderstandThatThisIsAPotentiallyDestructiveOperation=True)
        
        

if __name__ == '__main__':    
    unittest.main()
    # tbtest = TextbaseReadersTest()
    
    # test = EncoderTest()
    # test.testCaching()
    # import hashlib
    # sha1encoder = hashlib.sha1()