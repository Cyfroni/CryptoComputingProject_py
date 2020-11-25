import integer
import random


def funcL(x, n):
    return (x - 1) // n


def sampleGen(n):
    g = random.randint(1, n - 1)
    while integer.gcd(g, n) != 1:
        g = random.randint(1, n - 1)
    return g


def keyGen(sbit):
    p = integer.randprime(int(sbit/2))
    q = integer.randprime(int(sbit/2))
    while integer.gcd(p*q, (p-1)*(q-1)) != 1:
        p = integer.randprime(int(sbit/2))
        q = integer.randprime(int(sbit/2))
    n = p * q
    n2 = n * n

    lamb = integer.lcm(p - 1, q - 1)
    g = sampleGen(n2)
    while integer.gcd(funcL(integer.fast_pow(g, lamb, n2), n), n) != 1:
        g = sampleGen(n2)
    miu = integer.inverse(funcL(integer.fast_pow(g, lamb, n2), n), n)

    return n, g, lamb, miu


def encrypt(m, n, g):
    if m < 0 or m >= n:
        raise Exception("message m must be not less than 0 and less than n")

    r = random.randint(1, n - 1)
    n2 = int(n**2)
    while integer.gcd(r, n2) != 1:
        r = random.randint(1, n - 1)

    c = integer.mod(integer.fast_pow(g, m, n2) *
                    integer.fast_pow(r, n, n2), n2)
    return c


def decrypt(c, n, g, lamb, miu):
    n2 = n * n
    if integer.gcd(c, n2) != 1:
        print("error")
    if c < 1 or c >= n2 or integer.gcd(c, n2) != 1:
        raise Exception("cipher c must be in Group Z_*_n^2")
    m_bar = integer.mod(funcL(integer.fast_pow(c, lamb, n2), n) * miu, n)
    return m_bar


def plaintextAdd(c1, c2, n, g):
    n2 = n * n
    c_ = integer.mod(c1 * c2, n2)
    return c_


if __name__ == "__main__":
    # import time

    # print("Generating key pair...")
    n, g, l, m = keyGen(256)
    # print("Generated key pair.")
    plaintext1 = random.randint(5000, 10000)
    plaintext2 = random.randint(5000, 10000)
    # tstart = time.time()
    c1 = encrypt(plaintext1, n, g)
    c2 = encrypt(plaintext2, n, g)
    # tend = time.time()
    # print("average time: " + str((tend - tstart) / 2))
    c_ = plaintextAdd(c1, c2, n, g)
    # tstart = time.time()
    m1 = decrypt(c1, n, g, l, m)
    m2 = decrypt(c2, n, g, l, m)
    m_bar = decrypt(c_, n, g, l, m)
    # tend = time.time()
    # print("c1:    " + str(c1))
    # print("c2:    " + str(c2))
    # print("c_:    " + str(c_))
    print("m1:    " + str(m1))
    print("m2:    " + str(m2))
    print("m_bar: " + str(m_bar))
    # print("average time: " + str((tend - tstart) / 3))
