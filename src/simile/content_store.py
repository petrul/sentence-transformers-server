
class ContentStore:
            
    def put(self, content: str) -> int :
        id = ContentStore.id(content)
        return id
    
    def put(self, id, content: str):
        pass 
    
    def get(self, id) -> str:
        raise 'not implemented'
    
    def __getitem__(self, key):
        return self.get(key)
    
    @staticmethod
    def id(content: str) -> int: 
        return hash(content)


class DictContentStore(ContentStore):
    store: dict = {}
    
    def put(self, content: str) -> int:
        id = super().put()
        self.put(id, content)
        return id
    
    def put(self, id, content):
        self.store[id] = content
    
    def get(self, id):
        return self.store[id]
