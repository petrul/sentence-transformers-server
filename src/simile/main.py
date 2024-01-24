from content_store import *
from tb_readers import *
from vecstore import *

class Main:

    @staticmethod  
    def main():
        vs = VectorStoreFactory.inmemory()
        cs = DictContentStore()
        
        tbdl = TextbaseDownloads()
        
        model = EncoderFactory.all_MiniLM_L6_v2()
        
        n = 120
        paras = list(islice(tbdl.sentences(), n))
        
        p(len(paras))
        ids = [f"{it.path}/{it.location}" for it in paras]
        
        for id, ch in zip(ids, paras):
            txt = ch.content
            cs.put(id,txt)
            vect = model.encode(txt)
            vs.put(id, vect)
        
        p(f'nr of stored vecs: {len(vs)}')
        ref = ids[15]
        p("=" * 80)
        p(f'ref: {ref} ->\n{cs[ref]}')
        p("=" * 80)
        nearestK = vs.nearest(ids[0], 10)
        parr([f'{it.weight} - {it.id} - {cs[it.id]} (size: {len(cs[it.id])})'
              for it in nearestK])


def parr(arr):
    p('\n\n'.join(arr))

if __name__ == '__main__':
    Main.main()
    
    