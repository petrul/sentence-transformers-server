# textbase readers stuff
from dataclasses import dataclass
import os

@dataclass
class FilenameAndContent:
    path: str
    location: str # extra location within a path
    content: str
    
    def __init__(self, path, content, location=None):
        self.path = path
        self.content = content
        self.location = location
        
    def id(self, basepath: str):
        assert self.path.startswith(basepath)
        relevantPath =  self.path[len(basepath):]
        if self.location == None:
            return relevantPath
        return f'{relevantPath}#{self.location}'

@dataclass
class TextbaseDownloads:
    
    def __init__(self, basedir="~/data/textbase-dl/"):
        self.basedir = os.path.expanduser(basedir)
    
    # traverse root directory, and list directories as dirs and files as files
    def files(self) :
        for root, dirs, files in os.walk(self.basedir):
            for file in files:
                cmplPath = os.path.join(root, file)
                yield cmplPath
    

    # returns 2-tuples of (filepath, content)
    def chapters(self):     
        for f in self.files():
            with open(f) as file:
                txt = file.read()
                yield FilenameAndContent(path=f, content=txt)
                
    def paragraphs(self):        
        for ch in self.chapters():
            txt = ch.content
            paras = txt.split('\n')
            i = 0
            for p in paras:
                if p.strip() != "":
                    yield FilenameAndContent(path=ch.path, location="%d" % i, content=p)
                i += 1

    def sentences(self):
        for ch in self.paragraphs():
            txt = ch.content
            sentences = txt.split('.')
            i = 0
            for s in sentences:
                s = s.strip()
                if s != "" and len(s) > 10:
                    yield FilenameAndContent(path=ch.path, location="%s-%d" % (ch.location, i), content=s)
                i += 1
    
    @staticmethod
    def get_sentence(filenameId: str):
        assert "/" in filenameId
        assert "-" in filenameId
        split = filenameId.split("/")
        lastElem = split[-1]
        split = lastElem.split("-")
        paragraphNr, sentenceNr = split[0], split[1]
        return TextbaseDownloads.get_sentence()
    
    @staticmethod
    def get_sentence(filenamePath: str, paragraphNr: int, sentenceNr: int):
        with open(filenamePath) as file:
                txt = file.read()
                paras = txt.split('\n')
                para = paras[paragraphNr]
                sentences = para.split('.')
                sentence = sentences[sentenceNr]
                return sentence
        

def p(args): print(args)

if __name__ == "__main__":
    tbdl = TextbaseDownloads()
    p(tbdl.files())
    p(tbdl.chapters())
    paths = [it.path for it in tbdl.chapters()]
    p(len(paths))
    assert len(paths) > 0
