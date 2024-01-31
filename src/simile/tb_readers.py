# textbase readers stuff
from dataclasses import dataclass
import os
import pathlib

# @dataclass
# class FilenameAndContent:
#     path: str
#     location: str # extra location within a path
#     content: str
    
#     def __init__(self, path, content, location=None):
#         self.path = path
#         self.content = content
#         self.location = location
        
#     def id(self, basepath: str):
#         assert self.path.startswith(basepath)
#         relevantPath =  self.path[len(basepath):]
#         if self.location == None:
#             return relevantPath
#         return f'{relevantPath}#{self.location}'

class DlContent:
    def getId(self) -> str:
        pass
    def text(self) -> str:
        pass

@dataclass
class DlFile(DlContent):
    '''
        represents a dumped text file
    '''
    basedir: str
    dir: str
    filename: str
    _text: str = None # cache
    
    def getId(this) -> str:
        assert this.dir.startswith(this.basedir)
        specificPath = this.dir[len(this.basedir):]
        return f'{specificPath}/{this.filename}'
    
    def getCompletePath(this) -> str:
        return os.path.join(this.dir, this.filename)
    
    def getTextbaseUrl(self) -> str:
        import re
        id = self.filename
        id = id.replace('__', '/')
        id = id[:-4]
        id = re.sub('\d+', '|', id)
        id = id.replace('|_|_', '/')
        return f'https://textbase.scriptorium.ro{id}'
    

    def text(self) -> str:
        if self._text == None:
            cpath = self.getCompletePath()
            with open(cpath) as file:
                self._text = file.read()
        return self._text
    
    def paragraphs(self):
        txt = self.text()
        paras = txt.split('\n')
        for i, para in enumerate(paras):
            yield (i, para)
    

@dataclass
class DlParagraph(DlContent):
    file: DlFile
    index: int
    _text: str = None # text cache
    
    def getId(self) -> str:
        return f'{self.file.getId()}#{self.index}'
    

    def text(self) -> str:
        if self._text == None:
            txt = self.file.text()
            paras = txt.split('\n')
            self._text = paras[self.index]
        return self._text
    
    def sentences(self):
        txt = self.text()
        sentences = txt.split('.')
        for i, sent in enumerate(sentences):
            yield (i, sent)

    
@dataclass
class DlSentence(DlContent):
    paragraph: DlParagraph
    index: int
    _text: str = None
        
    def getId(self) -> str:
        return f'{self.paragraph.getId()}-{self.index}'
    
    def text(self) -> str:
        if self._text == None:
            txt = self.paragraph.text()
            self._text = txt.split('.')[self.index]
        return self._text
    
    @staticmethod
    def fromMilvusId(basedir: str, milvId: str) :
        if not '#' in milvId:
            raise Exception(f'id [{milvId}] should contain # character')
        
        filenameWithAnchor = os.path.basename(milvId)
        dirname = os.path.dirname(milvId)
        [filename, idxs] = filenameWithAnchor.split('#')
        assert '-' in idxs
        [idxPara, idxSentence] = idxs.split('-')
        dltext = DlFile(basedir, os.path.join(basedir,dirname), filename)
        para = DlParagraph(dltext, int(idxPara))
        dlsentence = DlSentence(para, int(idxSentence))
        return dlsentence

@dataclass
class TextbaseDownloads:
    '''
        models a download dump of textbase, which is really lot of .txt files with a 
        naming convention, of the whole Textbase site.
    '''
    basedir: str
    
    def __init__(self, basedir="~/data/textbase-dl/"):
        self.basedir = os.path.expanduser(basedir)
        assert pathlib.Path(self.basedir).is_dir()
    
    # traverse root directory, and list directories as dirs and files as files
    def files(self) :
        for root, dirs, files in os.walk(self.basedir):
            dirs.sort()
            for file in files:
                yield DlFile(self.basedir, root, file) # cmplPath
                
    def paragraphs(self):        
        for dfile in self.files():
            for i, para in dfile.paragraphs():
                if para.strip() != "":
                    yield DlParagraph(dfile, i, para)

    # you probably do not want to use this but rather significant_sentences() instead.
    def sentences(self):
        for para in self.paragraphs():
            for i, sentence in para.sentences():
                sentence = sentence.strip()
                yield DlSentence(para, i, sentence)
    
    def significant_sentences(self):
        for sentence in self.sentences():
            txt = sentence.text()
            words = [it for it in txt.split(' ') if it]
            if txt.strip() != "" and len(txt) > 7 and len(words) > 1:
                yield sentence
            
    
    # @staticmethod
    # def get_sentence(filenameId: str):
    #     assert "/" in filenameId
    #     assert "-" in filenameId
    #     split = filenameId.split("/")
    #     lastElem = split[-1]
    #     split = lastElem.split("-")
    #     paragraphNr, sentenceNr = split[0], split[1]
    #     return TextbaseDownloads.get_sentence()
    
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
    # for f in tbdl.files():
    #     p(f.getId())
    # quit()
    # p(tbdl.chapters())
    paths = [it.getCompletePath for it in tbdl.files()]
    p(len(paths))
    assert len(paths) > 0

    # quit()
    
    for sent in tbdl.significant_sentences():
        p(sent.getId())
        p(sent.paragraph.file.getCompletePath())
        # p(sent.paragraph.getId())
        # file = sent.paragraph.file
        # p(sent.paragraph.file.getId())
        # p(file.path)
        # p(file.getCompletePath)
        # p(f'{sent.paragraph.file.getId()} - {sent.index} - {len(sent.text())}')
