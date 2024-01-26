import os
import time

def scriptDir(filepath = __file__):
    return os.path.dirname(os.path.realpath(filepath))

def p(*args): print(*args)

def join(list: list[str], sep = '\n') -> str:
    return sep.join(list)

def randomAlphabetic(n = 10) -> str:
    import random, string
    resp = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(n))
    return resp


class StopWatch:
    
    def start(self):
        self.timestamp_start = time.time()
        return self
        
    def stop(self):
        self.timestamp_end = time.time()
        self.took = self.timestamp_end - self.timestamp_start
        return self.took
    
    def __str__(self) -> str:
        return str(".2f" % self.took)


if __name__ == '__main__':
    a = ['a', 'b', "cd"]
    print(a)