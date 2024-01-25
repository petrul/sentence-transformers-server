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
        assert nfiles > 0

        files = list(tbdl.files())
        # assert authors sorted alphabetically        
        i_alecsandri = [i for (i, it) in enumerate(files)   if it.filename == '0000_35030_alecsandri__lacrimioare.txt' ][0]
        i_eminescu = [i for (i, it) in enumerate(files)     if it.filename == '0007_36961_eminescu__proza_antuma__cezara__iv.txt' ][0]
        assert i_alecsandri < i_eminescu
        
        nparas = sum(1 for it in tbdl.paragraphs())
        assert(nparas > nfiles)
        
        nsents = sum(1 for it in tbdl.sentences())
        assert(nsents > nparas)
        
        nss = sum(1 for it in tbdl.significant_sentences())
        assert (nss < nsents)
        assert nss > 0


    def testSignificantSentences(self):
        # tbdl = TextbaseDownloads()
        tbdl = TextbaseDownloads(self.textbase_downloads_dir)
        
        bufsize = 10
        # sents = tbdl.significant_sentences()
        for dls in tbdl.significant_sentences():
            assert len(dls.text()) > 5
            
    
    
    def testIdsStartWithAuthors(self):
        tbdl = TextbaseDownloads(self.textbase_downloads_dir)
        sentences = tbdl.sentences()
        for s in sentences:
            assert s.getId().startswith('/eminescu') or s.getId().startswith('/alecsandri')
        


if __name__ == '__main__':    
    tbtest = TextbaseReadersTest()
    # tbtest.testCountFiles()
    # tbtest.testSignificantSentences()
    tbtest.testIdsStartWithAuthors()