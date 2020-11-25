import random

n = 3
p = 97


def randint():
    return random.randint(0, p - 1)


def generate_shares(val):
    r = [randint() for _ in range(n - 1)]
    return [(val - sum(r)) % p] + r


def calc_MAC(key, a):
    alpha, beta = key
    return (alpha * a + beta) % p


class Offline:

    @staticmethod
    def opening(a_s, j):
        a = 0
        for a_, a_m in a_s:
            key, mac = a_m[j]

            assert calc_MAC(key, a_) == mac

            a += a_
        return a % p

    @staticmethod
    def addition(a_s, b_s):
        c_s = []
        for (a_, a_m), (b_, b_m) in zip(a_s, b_s):
            c_ = (a_ + b_) % p
            c_m = []
            for (a_mk, a_mm), (b_mk, b_mm) in zip(a_m, b_m):
                assert a_mk[0] == b_mk[0]  # check consistency
                c_mk = (a_mk[0], (a_mk[1] + b_mk[1]) % p)
                c_mm = (a_mm + b_mm) % p
                c_m.append((c_mk, c_mm))
            c_s.append((c_, c_m))

        return c_s

    @staticmethod
    def mult_const(a_s, c):
        r_s = []
        for (a_, a_m) in a_s:
            r_ = (a_ * c) % p
            r_m = []
            for (a_mk, a_mm) in a_m:
                r_mk = (a_mk[0], (a_mk[1] * c) % p)
                r_mm = (a_mm * c) % p
                r_m.append((r_mk, r_mm))
            r_s.append((r_, r_m))

        return r_s

    @staticmethod
    def addition_const(a_s, c):
        r_s = []
        for i, (a_, a_m) in enumerate(a_s):
            r_ = (a_ + c) % p if i == 0 else a_
            r_m = []
            for j, (a_mk, a_mm) in enumerate(a_m):
                r_mk = (
                    a_mk[0],
                    (a_mk[1] - c * a_mk[0]) % p if i == 0 else a_mk[1]
                )
                r_mm = a_mm
                r_m.append((r_mk, r_mm))
            r_s.append((r_, r_m))

        return r_s


def multiply(x_s, y_s, triple, j):
    a_s, b_s, c_s = triple

    epsilon_s = Offline.addition(x_s, Offline.mult_const(a_s, -1))
    epsilon = Offline.opening(epsilon_s, j)
    rho_s = Offline.addition(y_s, Offline.mult_const(b_s, -1))
    rho = Offline.opening(rho_s, j)

    z0_s = Offline.addition(c_s, Offline.mult_const(b_s, epsilon))
    z1_s = Offline.addition_const(
        Offline.mult_const(a_s, rho),
        (rho * epsilon) % p
    )

    return Offline.addition(z0_s, z1_s)


# a -> a
# a_s -> [a]
# a_ -> ai
# a_m -> [(Kia0, m0(ai)), (Kia1, m1(ai)), ...]
# a_mk -> Kiaj
# a_mm -> mj(ai)

# a_s = [ ..., (a_, a_m), ... ]
# a_m = [ ..., (a_mk, a_mm), ... ]


class Party:
    def __init__(self):
        self.alphas = [randint() for _ in range(n)]

    def singles(self, a=randint()):
        a_s = generate_shares(a)
        messages = []
        for i in range(n):
            keys = [(alpha, randint()) for alpha in self.alphas]
            macs = [calc_MAC(key, a_s[i]) for key in keys]
            messages.append([
                a_s[i],
                list(zip(keys, macs))
            ])
        return messages

    def triples(self):
        a = randint()
        b = randint()
        c = (a * b) % p
        return [self.singles(a), self.singles(b), self.singles(c)]


def test_offline():
    party = Party()

    # opening
    print(f"opening_test")
    a = randint()
    a_s = party.singles(a)

    for i in range(n):
        assert a == Offline.opening(a_s, i)
    print(a, "\n")

    # addition
    print(f"addition_test")
    a = randint()
    b = randint()
    c = (a + b) % p
    a_s = party.singles(a)
    b_s = party.singles(b)
    c_s = Offline.addition(a_s, b_s)

    for i in range(n):
        assert c == Offline.opening(c_s, i)
    print(f"[{a}] + [{b}] mod {p} = [{c}]\n")

    # mult_const
    print(f"mult_const_test")
    a = randint()
    b = randint()
    r = (a * b) % p
    a_s = party.singles(a)
    r_s = Offline.mult_const(a_s, b)

    for i in range(n):
        assert r == Offline.opening(r_s, i)
    print(f"[{a}] * {b} mod {p} = [{r}]\n")

    # addition_const
    print(f"addition_const_test")
    a = randint()
    b = randint()
    r = (a + b) % p
    a_s = party.singles(a)
    r_s = Offline.addition_const(a_s, b)

    for i in range(n):
        assert r == Offline.opening(r_s, i)
    print(f"[{a}] + {b} mod {p} = [{r}]\n")


def test_online():
    party = Party()
    triple = party.triples()

    # mult
    print("mult_test")
    x = randint()
    y = randint()
    z = (x * y) % p
    x_s = party.singles(x)
    y_s = party.singles(y)

    for i in range(n):
        z_s = multiply(x_s, y_s, triple, i)
        for j in range(n):
            assert z == Offline.opening(z_s, j)

    print(f"[{x}] * [{y}] mod {p} = [{z}]\n")


if __name__ == "__main__":
    test_offline()
    test_online()


# print(*partys[0].triples(), sep="\n")
