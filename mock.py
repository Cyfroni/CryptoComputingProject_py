'''

## print ##

def print(*args, **kwargs): pass


## paillier ##

paillier.encrypt = lambda m, *args: m
paillier.decrypt = lambda c, *args: c

'''
