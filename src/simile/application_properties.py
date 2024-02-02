import os

class ApplicationProperties:
    
    def cacheDir(self) -> str:
        KEY = 'SENTENCE_TRANSFORMERS_SERVER_CACHE_DIR'
        if KEY in os.environ.keys():
            return os.environ[KEY]
        
        return os.path.expanduser('~/.sentence_transformers_server/cache')