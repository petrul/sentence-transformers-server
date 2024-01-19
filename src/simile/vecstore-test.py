# from 'store'

import unittest
from vecstore import *

class StoreTest(unittest.TestCase):
            
    def testGetPut(self):
        encoder = EncoderFactory.all_MiniLM_L6_v2()    
        sentences = [
            'the dog barks',
            'the dog is brown',
            'the cat meaws',
            'the sun burns hot today',
            # 'câinele latră'
        ]
        ids = ["id%d" % i for i in range(len(sentences))]
        
        embeddings = encoder.encode(sentences)
        vs = VectorStoreFactory.inmemory()
        assert len(vs) == 0
        id0 = ids[0]
        emb0 = embeddings[0]
        vs.put(id0, emb0)
        # p(vs.get(id0))
        assert len(vs) == 1
        assert emb0 is vs.get(id0)
        

    def testInMemoryStore(self):
        encoder = EncoderFactory.all_MiniLM_L6_v2()    

        sentences = [
            'the dog barks',
            'the dog is brown',
            'the cat meaws',
            'the sun burns hot today',
            # 'câinele latră'
        ]
        ids = ["id%d" % i for i in range(len(sentences))]
        [id0, id1, id2, id3] = ids
        embeddings = encoder.encode(sentences)
        vecstore = VectorStoreFactory.inmemory()
        assert id0 == 'id0'
        emb0 = embeddings[0]
        # p(type(emb0))
        v0 = vecstore.put(id0, emb0)
        assert v0 is emb0
        againEmb0 = vecstore.get(id0)
        assert len(emb0) == len(againEmb0)
        for i in range(len(emb0)):
            assert emb0[i] == againEmb0[i]

        vecstore.put(ids[1], embeddings[1])
        vecstore.put(ids[2], embeddings[2])
        vecstore.put(ids[3], embeddings[3])

        assert len(vecstore) == 4
        
        dist01 = vecstore.distance(id0, id1)
        assert dist01 > 0
        assert vecstore.distance(id0, id2) > 0
        assert vecstore.distance(id0, id3) > 0
        assert vecstore.distance(id1, id2) > 0
        assert vecstore.distance(id1, id3) > 0
        assert vecstore.distance(id2, id3) > 0
        
        nearest_all = vecstore.nearest(id0)
        assert len(nearest_all) == 3
        
        nearest2 = vecstore.nearest(id0, 2)
        assert len(nearest2) == 2
        for elem in nearest2:
            assert elem.weight  in [it.weight for it in nearest_all]
            assert elem.id      in [it.id for it in nearest_all]
            assert elem.index   in [it.index for it in nearest_all]    
        

def p(*args):
    print(*args)

if __name__ == '__main__':
    unittest.main()
    # StoreTest().testGetPut()
    # StoreTest().testInMemoryStore()
    
    
    
