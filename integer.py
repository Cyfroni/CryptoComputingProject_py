import random


def mod(a, n):
    return int(a % n)


def fast_pow(g, a, p):
    e = mod(a, p - 1)
    if e == 0:
        return 1
    import math
    r = int(math.log2(e))  # + 1 - 1
    x = g
    for i in range(0, r):
        x = mod(x**2, p)
        if (e & (1 << (r - 1 - i))) == (1 << (r - 1 - i)):
            x = mod(g * x, p)
    return int(x)


def isPrime_MR(u, T):
    # calculate v and w , let u - 1 = w * 2^v
    v = 0
    w = u - 1
    while mod(w, 2) == 0:
        v += 1
        w = w // 2
    for _ in range(1, T + 1):
        nextj = False
        a = random.randint(2, u - 1)
        b = fast_pow(a, w, u)
        if b == 1 or b == u - 1:
            nextj = True
            continue
        for _ in range(1, v):
            b = mod(b**2, u)
            if b == u - 1:
                nextj = True
                break
            if b == 1:
                return False
        if not nextj:
            return False
    return True


def randprime(bitlen):
    lowbound = (1 << bitlen) + 1
    upbound = (1 << (bitlen + 1)) - 1
    while(True):
        rint = random.randint(lowbound, upbound)
        if mod(rint, 2) == 1 and isPrime_MR(rint, 15):
            return rint


def swap(a, b):
    return b, a


def is_even(a):
    if (mod(a, 2)) == 0:
        return True
    else:
        return False


def gcd(a, b):
    if a < 0:
        a = -a
    if b < 0:
        b = -b
    if b == 0:
        if a != 0:
            a, b = swap(a, b)
        else:
            return 0
    k = 0
    while is_even(a) and is_even(b):
        a = a >> 1
        b = b >> 1
        k += 1
    if is_even(b):
        a, b = swap(a, b)
    while True:
        while is_even(a):
            a = a >> 1
        if a < b:
            a, b = swap(a, b)
        a = (a - b) >> 1
        if a == 0:
            d = int(b * (2**k))
            return d


def lcm(a, b):
    return (a * b) // gcd(a, b)


def inverse(a, n):
    s, old_s = 0, 1
    t, old_t = 1, 0
    r, old_r = n, a

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    return mod(old_s, n)
