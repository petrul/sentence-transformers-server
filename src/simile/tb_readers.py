# textbase readers stuff
from dataclasses import dataclass

@dataclass
class FilenameAndContent:
    path: str
    location: str # extra location within a path
    content: str
    
    def __init__(self, path, content, location=None):
        self.path = path
        self.content = content
        self.location = location

@dataclass
class TextbaseDownloads:
    
    basedir = "/home/petru/data/textbase-dl/dickens"
    
    # traverse root directory, and list directories as dirs and files as files
    def files(self) :
        import os
        for root, dirs, files in os.walk(self.basedir):
            # path = root.split(os.sep)
            # p(path)
            # print((len(path) - 1) * '---', os.path.basename(root))
            for file in files:
                # print(len(path) * '---', file)
                cmplPath = os.path.join(root, file)
                yield cmplPath
    

    # returns 2-tuples of (filepath, content)
    def chapters(self):     
        for f in self.files():
            with open(f) as file:
                txt = file.read()
                yield FilenameAndContent(path=f, content=txt)
                
    def paragraphs(self):
        i = 0
        for ch in self.chapters():
            txt = ch.content
            paras = txt.split('\n')
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
                s = s.strip() + '.'
                if s != ".":
                    yield FilenameAndContent(path=ch.path, location="%s-%d" % (ch.location, i), content=s)
                i += 1

def p(args): print(args)

if __name__ == "__main__":
    tbdl = TextbaseDownloads()
    p(tbdl.files())
    p(tbdl.chapters())
    paths = [it.path for it in tbdl.chapters()]
    p(len(paths))
    assert len(paths) > 0
