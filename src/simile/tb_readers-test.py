import unittest
from tb_readers import *
from util import *
from itertools import islice

class TextbaseReadersTest(unittest.TestCase):
    textbase_downloads_dir = os.path.abspath(f'{scriptDir()}/../../tests/resources/textbase-dl')
    
            
    def testCountFiles(self):
        # tbdl = TextbaseDownloads()
        tbdl = TextbaseDownloads(self.textbase_downloads_dir)
    
    
        nfiles = sum(1 for it in tbdl.files())
        p(nfiles)
        assert nfiles > 0

        files = list(tbdl.files())
        # assert authors sorted alphabetically        
        i_alecsandri = [i for (i, it) in enumerate(files)   if it.filename == '0000_35030_alecsandri__lacrimioare.txt' ][0]
        i_eminescu = [i for (i, it) in enumerate(files)     if it.filename == '0007_36961_eminescu__proza_antuma__cezara__iv.txt' ][0]
        assert i_alecsandri < i_eminescu
        
        nparas = sum(1 for it in tbdl.paragraphs())
        p(nparas)
        assert(nparas > nfiles)
        
        nsents = sum(1 for it in tbdl.sentences())
        p(nsents)
        assert(nsents > nparas)
        
        nss = sum(1 for it in tbdl.significant_sentences())
        p(nss)
        assert (nss < nsents)
        assert nss > 0
        
        # sents = [it.text() for it in tbdl.sentences()]
        # p('gata to sents')
        # ssents = [it.text() for it in tbdl.significant_sentences()]
        # p('gata to ssents')
        # p([it for it in sents if not it in ssents])
        # p(join([it for it in ssents]))


    def testSignificantSentences(self):
        # tbdl = TextbaseDownloads()
        tbdl = TextbaseDownloads(self.textbase_downloads_dir)
        
        bufsize = 10
        # sents = tbdl.significant_sentences()
        for dls in tbdl.significant_sentences():
            assert len(dls.text()) > 5
            
        
        
        # while True:
        #     buff = list(islice(sents, bufsize))
        #     if len(buff) == 0: break
            
        #     p('\n==\n'.join([it.text() for it in buff]))
        #     input()
        

        
        
        
        
        
if __name__ == '__main__':    
    tbtest = TextbaseReadersTest()
    tbtest.testCountFiles()
    # tbtest.testSignificantSentences()
        
