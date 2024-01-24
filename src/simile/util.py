import os

def scriptDir(filepath = __file__):
    return os.path.dirname(os.path.realpath(filepath))

def p(*args): print(*args)

def join(list: list[str], sep = '\n') -> str:
    return sep.join(list)

def randomAlphabetic(n = 10) -> str:
    import random, string
    resp = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(n))
    return resp