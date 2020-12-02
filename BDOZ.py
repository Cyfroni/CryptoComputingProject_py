import random
import paillier
from time import sleep
from party import Party, parties_init

paillier.encrypt = lambda m, *args: m
paillier.decrypt = lambda c, *args: c

n = 3
p = 97


def randint():
    return random.randint(0, p - 1)


def randints(u):
    return [randint() for _ in range(u)]


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


class Env:

    def __init__(self, n):
        self.parties = parties_init(n, BDOZParty)
        sleep(1)

    def share(self, u):
        for party in self.parties:
            xs = party.input3 if party.input3 else randints(u)
            xs_en = [party._encrypt(x) for x in xs]
            party._broadcast(xs_en)

            party.data["shares"] = (xs, xs_en)

        sleep(1)

        for party in self.parties:
            data = party._receive_broadcast(party.data["shares"][1])
            xk = list(zip(*data))

            party.output = xk

    def mult_2(self, u, i, j):
        # a = [ak[i] for ak in self.parties[i].input[:u]]
        # b = [bk[j] for bk in self.parties[i].input[u:]]

        a = self.parties[i].input2[:u]
        b = self.parties[i].input2[u:]

        rs = randints(u)
        C = [(self.parties[i].data["shares"][0][k] * b[k] + self.parties[i]._encrypt(rs[k], self.parties[j].pk))
             for k in range(u)]
        # print(C)

        self.parties[i]._unicast(self.parties[j].partyId, C)
        self.parties[i].output = [-r % p for r in rs]

        sleep(1)

        vs = self.parties[j]._receive_unicast(self.parties[i].partyId)
        self.parties[j].output = [self.parties[j]._decrypt(v) % p for v in vs]

    def mult_n(self, u):
        self.share(2 * u)
        for party in self.parties:
            party.input = party.output
            party.data['c'] = [
                (ak[party.partyId] * bk[party.partyId]) % p for ak, bk in zip(party.input[:u], party.input[u:])
            ]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                self.parties[i].input2 = self.parties[j].input2 = \
                    [ak[i] for ak in self.parties[i].input]

                self.mult_2(u, i, j)

                self.parties[i].data['c'] = [
                    (c + z) % p for c, z in zip(self.parties[i].data['c'], self.parties[i].output)]
                # print(self.parties[i].output)

                self.parties[j].data['c'] = [
                    (c + z) % p for c, z in zip(self.parties[j].data['c'], self.parties[j].output)]
                # print(self.parties[j].output)

        for i in range(n):
            self.parties[i].input3 = self.parties[i].data['c']

        self.share(u)

        # for k in range(u):
        #     print((self.parties[0].data['c'][k] + self.parties[1].data['c']
        #            [k] + self.parties[2].data['c'][k]) % p)

    def add_macx(self, u):

        for party in self.parties:
            party.data["alpha"] = randints(n)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                alpha = self.parties[i].data["alpha"][j]

                alpha_enc = self.parties[i]._encrypt(alpha)

                shares = self.parties[i].data["shares"][0]

                shares_enc = [
                    self.parties[i]._encrypt(share, self.parties[j].pk) for share in shares
                ]

                self.parties[i].input2 = self.parties[j].input2 = \
                    [alpha_enc] * u + self.parties[i].data["shares"][0]

                self.mult_2(u, i, j)

                bethas = [-r % p for r in self.parties[i].output]

                self.parties[i].data[f'mac{j}'] = \
                    (shares[j], (alpha, bethas))

                self.parties[j].data[f'mac2{i}'] = \
                    (shares[j], self.parties[j].output)
                # print(self.parties[i].output)

                # self.parties[j].data['c'] = [
                #     (c + z) % p for c, z in zip(self.parties[j].data['c'], self.parties[j].output)]
                # print(self.parties[j].output)

        # a -> a
        # a_s -> [a]
        # a_ -> ai
        # a_m -> [(Kia0, m0(ai)), (Kia1, m1(ai)), ...]
        # a_mk -> Kiaj
        # a_mm -> mj(ai)

        # a_s = [ ..., (a_, a_m), ... ]
        # a_m = [ ..., (a_mk, a_mm), ... ]


class BDOZParty(Party):
    def __init__(self, *args):
        self.sk = paillier.keyGen(10)  # n, g, lamb, miu
        self.pk = [self.sk[0], self.sk[1]]
        self.alphas = [randint() for _ in range(n)]
        self.data = {}
        self.output = []
        self.input = []
        self.input2 = []
        self.input3 = []
        super().__init__(*args)

    def __repr__(self):
        return f"""
        ## {self.partyId} ##
        data: {self.data}
        output: {self.output}
        input: {self.input}
        input2: {self.input2}
        input3: {self.input3}
        """

    def _get_messages(self):
        messages = self.serv.received_data
        self.serv.received_data = {}
        return messages

    def _get_message(self, source_id):
        message = self.serv.received_data[str(source_id)]
        self.serv.received_data[str(source_id)] = None
        return message

    def _to_message(self, vals):
        message = " ".join(map(str, vals))
        return [bytes(f"0 {self.partyId} {message}", 'ascii')]

    def _to_vals(self, message):
        return list(map(int, message))

    def _broadcast(self, vals):
        print(f"{self.partyId}: broadcast {vals}")
        self.broadcast_message(self._to_message(vals))

    def _receive_broadcast(self, vals):
        messages = self._get_messages()
        print(f"{self.partyId}: received {messages}")

        ret = [0] * n

        for party_id, message in messages.items():
            ret[int(party_id)] = self._to_vals(message)

        ret[self.partyId] = vals

        return ret

    def _unicast(self, target_id, vals):
        print(f"{self.partyId} -> {target_id}: unicast {vals}")
        self.unicast_message(target_id, self._to_message(vals))

    def _receive_unicast(self, source_id):
        message = self._get_message(source_id)
        print(f"{self.partyId}: received {message}")

        return self._to_vals(message)

    def _encrypt(self, m, key=None):
        key = key if key else self.pk
        return paillier.encrypt(m, *key)

    def _decrypt(self, c):
        return paillier.decrypt(c, *self.sk)

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
    env = Env(n)
    # env.share(5)
    env.mult_n(4)
    env.add_macx(4)
    # test_offline()
    # test_online()
    print(env.parties)


# print(*partys[0].triples(), sep="\n")
