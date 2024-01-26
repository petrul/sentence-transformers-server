import unittest
import util
from util import p
import encoder

class EncoderTest(unittest.TestCase):
    import tempfile
    import os
    
    tmp_dir = os.path.join(tempfile.gettempdir(), 'TextbaseReadersTest_' + util.randomAlphabetic(3))
    res_dir = os.path.abspath(f'{util.scriptDir()}/../../tests/resources/textbase-dl')
            
    def testCaching(self):
        
        cache: encoder.VectorCache = encoder.VectorCache(self.tmp_dir)
        enc = encoder.EncoderFactory.all_MiniLM_L6_v2(cache=cache)
        
        sentences = [
            "foaie verde",
            "gigea",
            "ma cam doare capul",
            "ma cam doare capul",
        ]
        
        resp1 = enc.encode(sentences)
        
        import os
        assert os.path.exists(os.path.join(cache.cachedir, '66/d3/17/66d317cc6ddc5cccc2cf0a350a33fcb2ac4c9842'))
        
        resp2 = enc.encode(sentences)
        
        assert len(resp1) == len(resp2)
        assert len(resp1) == len(sentences)
        
        for i, _ in enumerate(resp1):
            resp1[i] == resp2[i]
        
        cache.rm_rf_cachedir(iUnderstandThatThisIsAPotentiallyDangerousOperation=True)
        
        

if __name__ == '__main__':    
    # tbtest = TextbaseReadersTest()
    
    test = EncoderTest()
    test.testCaching()
    # import hashlib
    # sha1encoder = hashlib.sha1()
    # sha1encoder.update('foaie  verde1'.encode('utf-8'))
    # p(sha1encoder.hexdigest())