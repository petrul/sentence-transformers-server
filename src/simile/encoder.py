from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass
from typing import List

def p(*args): print(*args)

@dataclass
class Encoder:
    name: str
    st: SentenceTransformer
    
    def encode(self, sentences: str| list[str], **args):
        return self.st.encode(sentences=sentences, **args)

class EncoderFactory: 

    @staticmethod
    def all_MiniLM_L6_v2():
        name = 'all-MiniLM-L6-v2'
        resp = SentenceTransformer(name)
        p('model [%s], with max_seq_length %d' % (name, resp.get_max_seq_length()))        
        return Encoder(name, resp)    
