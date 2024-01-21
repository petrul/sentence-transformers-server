from sentence_transformers import SentenceTransformer, util
from itertools import islice
from dataclasses import dataclass
import numpy

def p(*args):
    print(*args)

# use this to hold an id with the weight
@dataclass
class WeightedId:
    id: object  # the id of the object being weighted, can be a string, a url
    index: int  # internal index, an int, in the internal matrix used to store distances
    weight: float # the actual weight.


class Store:
    
    def put(self, key: object, value: numpy.ndarray[float] | list[float] ):
        pass
    
    def get(self, key, value) -> object:
        pass
    
    def nearest(self, key, limit) -> list[WeightedId]: 
        'get the nearest n to the document represented by the given key'
        pass
    
    def query(self, query: str, limit) -> list:
        '''
            query is similar to nearest but its content is given as is 
            rather than an existing document
            
            @param query should be an embedding.
            
        '''
        pass


class EncoderFactory: 

    @staticmethod
    def all_MiniLM_L6_v2():
        name = 'all-MiniLM-L6-v2'
        resp = SentenceTransformer(name)
        p('model [%s], with max_seq_length %d' % (name, resp.get_max_seq_length()))
        return resp

class VectorStoreFactory:
        
    @staticmethod
    def inmemory() -> Store: 
        return InMemoryVectorStore()

        
class InMemoryVectorStore(Store):
    '''in memory vector store'''
    vectors = {}
    size = 0 # the actual number of stored vectors
    
    id2idx = {} # map the distance line index (i) for a given id
    # the distance 
    
    incFactor = 2 
        
    def __init__(self, initMaxSize = 1000):
        self.initMaxSize = initMaxSize
        self.maxSize = initMaxSize
        self.distances = numpy.ndarray(shape=(initMaxSize, initMaxSize))
        
    def __len__(self) -> int:
        return self.size

    # puts the value and also makes up an id for it.
    def put(self, value) -> int : 
        id = self.size
        self.put(id, value)
        return id    

    def put(self, key: str, value):
        i = self.size #  this will be the line of this vector
        if key in self.id2idx :
            i = self.id2idx[key]
        
        assert(type(value) == numpy.ndarray)
        
        self.vectors[key] = value
        self.id2idx[key] = i
        
        # compute distances to all existing
        for id in self.vectors:
            vect = self.vectors[id]
            j_idx = self.id2idx[id]
            dist = InMemoryVectorStore.compute_distance(value, vect)
            self.distances[i, j_idx] = dist
            self.distances[j_idx, i] = dist # symmetrical

        self.size += 1
        return value
        
        
    def get(self, key):
        return self.vectors[key]
    
    # get the nearest  to the document represented by the given key
    def nearest(self, key, n = 0) -> list[WeightedId]:
        
        if (n == 0) :
            n = self.size - 1 # return all
        
        otherIds = filter(lambda it: it != key, self.vectors.keys())
        
        weigtedTerms = [
            WeightedId(it, self.id2idx[it], self.distance(key, it)) 
            for it in otherIds
        ]
        
        firstN = weigtedTerms[:n]
        sortedTerms = sorted(firstN, key=lambda it: it.weight, reverse=True)
        return sortedTerms
        # sortedFirstNIds = [it.id for it in sortedTerms]
        # sortedTerms = [it) for it in sortedFirstNIds]
        # return sortedVecs
    
    def distance(self, id1, id2):
        # v1 = self.get(id1)
        # v2 = self.get(id2)
        # return util.cos_sim(v1, v2)
        i = self.id2idx[id1]
        j = self.id2idx[id2]
        return self.distances[i][j]

    @staticmethod
    def compute_distance(vect1: numpy.ndarray, vect2: numpy.ndarray):
        return util.cos_sim(vect1, vect2)


if __name__ == "__main__":
    pass
    # testInMemoryStore()