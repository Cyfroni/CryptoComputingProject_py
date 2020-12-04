import random
import paillier
import re
import math
from time import sleep
from party import Party, parties_init

# paillier.encrypt = lambda m, *args: m
# paillier.decrypt = lambda c, *args: c

sbit = 32
primes = paillier.genPrimes(sbit)
modulo = primes[0] * primes[0] * primes[1] * primes[1]

n = 3
p = primes[0] * primes[1]


def reg_print(arg):
    limit = math.ceil(math.log10(p)) + 1
    reg = "\d{" + str(limit) + ",}"

    arg = re.sub(reg, "**", str(arg))

    print(arg)


def randint():
    return random.randint(0, p - 1)


def randints(u):
    return [randint() for _ in range(u)]


'''
def generate_shares(val):
    r = [randint() for _ in range(n - 1)]
    return [(val - sum(r)) % p] + r


def calc_MAC(key, a):
    alpha, beta = key
    return (alpha * a + beta) % p


class Offline:
    # a -> a
    # a_s -> [a]
    # a_ -> ai
    # a_m -> [(Kia0, m0(ai)), (Kia1, m1(ai)), ...]
    # a_mk -> Kiaj
    # a_mm -> mj(ai)

    # a_s = [ ..., (a_, a_m), ... ]
    # a_m = [ ..., (a_mk, a_mm), ... ]

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
'''


class Env:

    def __init__(self, n):
        self.parties = parties_init(n, BDOZParty)
        sleep(1)

    def _forward(self):
        for party in self.parties:
            party.input = party.output

    def _clear(self):
        for party in self.parties:
            party.clear()

    def _publish(self):
        for party in self.parties:
            party.triples.extend(party.output)

    def share(self, u):
        for party in self.parties:
            xki = party.input_share if party.input_share else randints(u)
            Exki = [party._encrypt(x) for x in xki]
            party._broadcast(Exki)

            party.data["shares"] = (xki, Exki)

        sleep(0.1)

        for party in self.parties:
            data = party._receive_broadcast(party.data["shares"][1])

            party.output = list(zip(*data))

    def mult_2(self, u, partyi, partyj):
        a = partyi.input_mult_2
        b = partyj.input_mult_2

        rs = randints(u)

        C = [
            (partyi._decrypt(a[k]) * b[k] + partyi._encrypt(rs[k], partyj.pk)) % modulo for k in range(u)
        ]

        partyi._unicast(partyj.partyId, C)
        partyi.output = [-r % p for r in rs]

        sleep(0.1)

        vs = partyj._receive_unicast(partyi.partyId)

        partyj.output = [partyj._decrypt(v) % p for v in vs]

    def mult_n(self, u):
        for party in self.parties:
            party.data['<ak>'] = party.input[:u]
            party.data['<bk>'] = party.input[u:]

            party.data['ak'] = [
                ak[party.partyId] for ak in party.data['<ak>']
            ]
            party.data['bk'] = [
                bk[party.partyId] for bk in party.data['<bk>']
            ]

            party.data['~ck'] = [
                (party._decrypt(aki) * party._decrypt(bki)) %
                p for aki, bki in zip(party.data['ak'], party.data['bk'])
            ]

        for partyi in self.parties:
            for partyj in self.parties:
                if partyi.partyId == partyj.partyId:
                    continue

                partyi.input_mult_2 = partyi.data['ak']
                partyj.input_mult_2 = partyj.data['bk']

                self.mult_2(u, partyi, partyj)

                for party in [partyi, partyj]:
                    party.data['~ck'] = [
                        (c + z) % p for c, z in zip(party.data['~ck'], party.output)
                    ]

        for party in self.parties:
            party.input_share = party.data['~ck']

        self.share(u)

    def add_macs(self, u):
        for party in self.parties:
            party.data["<ak>"] = party.input

            party.data["ak"] = [
                ak[party.partyId] for ak in party.data['<ak>']
            ]

            party.data["alpha"] = randints(n)
            party.data["alpha"][party.partyId] = None

            party.data['~[ak]'] = [
                (party._decrypt(ak), [[], []]) for ak in party.data["ak"]
            ]

        for partyi in self.parties:
            for partyj in self.parties:
                if partyi.partyId == partyj.partyId:
                    continue

                alpha = partyi.data["alpha"][partyj.partyId]

                alpha_enc = partyi._encrypt(alpha)

                partyi.input_mult_2 = [alpha_enc] * u
                partyj.input_mult_2 = partyj.data["ak"]

                self.mult_2(u, partyi, partyj)

                betas = [-r % p for r in partyi.output]
                keys = [(alpha, beta) for beta in betas]
                macs = partyj.output

                for k in range(u):
                    partyi.data['~[ak]'][k][1][0].append(keys[k])
                    partyj.data['~[ak]'][k][1][1].append(macs[k])

        for party in self.parties:
            party.output = party.data['~[ak]']

    def singles(self, u):

        self.share(u)

        self._forward()

        self.add_macs(u)

    def triples(self, u):
        self.share(4 * u)

        for party in self.parties:
            party.data["<ak>"] = party.output[:u]
            party.data["<bk>"] = party.output[u:(2*u)]
            party.data["<fk>"] = party.output[(2*u):(3*u)]
            party.data["<gk>"] = party.output[(3*u):]

            party.input = party.data["<ak>"] + party.data["<bk>"]

        self.mult_n(u)

        for party in self.parties:
            party.data["<ck>"] = party.output
            party.input = party.data["<fk>"] + party.data["<gk>"]

        self.mult_n(u)

        for party in self.parties:
            party.data["<hk>"] = party.output

            party.input = party.data["<ak>"] + party.data["<bk>"] + party.data["<ck>"] + \
                party.data["<fk>"] + party.data["<gk>"] + party.data["<hk>"]

        self.add_macs(6 * u)

        for party in self.parties:
            party.data["[ak]"] = party.output[:u]
            party.data["[bk]"] = party.output[u:(2*u)]
            party.data["[ck]"] = party.output[(2*u):(3*u)]
            party.data["[fk]"] = party.output[(3*u):(4*u)]
            party.data["[gk]"] = party.output[(4*u):(5*u)]
            party.data["[hk]"] = party.output[(5*u):]

            # TODO: check

            party.output = [
                (a, b, c) for a, b, c in zip(
                    party.data["[ak]"],
                    party.data["[bk]"],
                    party.data["[ck]"]
                )
            ]


class BDOZParty(Party):

    def __init__(self, *args):
        self.sk = paillier.keyGen(sbit, primes)  # n, g, lamb, miu
        self.pk = [self.sk[0], self.sk[1]]
        self.data = {}
        self.vars = []
        self.triples = []
        self.input_share = None
        self.input_mult_2 = None
        self.input = None
        self.output = []
        super().__init__(*args)

    def __repr__(self):
        return f"""


        ## {self.partyId} ##
        data: {self.data}

        input_share: {self.input_share}
        input_mult_2: {self.input_mult_2}
        input: {self.input}
        output: {self.output}

        vars({len(self.vars)}): {self.vars}
        triples({len(self.triples)}): {self.triples}
        """

    def full_clear(self):
        self.vars = []
        self.triples = []
        self.output = []
        self.clear()

    def clear(self):
        self.data = {}
        self.input_share = None
        self.input_mult_2 = None
        self.input = None

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
        reg_print(f"{self.partyId}: broadcast {vals}")
        self.broadcast_message(self._to_message(vals))

    def _receive_broadcast(self, vals):
        messages = self._get_messages()
        # reg_print(f"{self.partyId}: received {messages}")

        ret = [0] * n

        for party_id, message in messages.items():
            ret[int(party_id)] = self._to_vals(message)

        ret[self.partyId] = vals

        return ret

    def _unicast(self, target_id, vals):
        reg_print(f"{self.partyId} -> {target_id}: unicast {vals}")
        self.unicast_message(target_id, self._to_message(vals))

    def _receive_unicast(self, source_id):
        message = self._get_message(source_id)
        # reg_print(f"{self.partyId}: received {message}")

        return self._to_vals(message)

    def _encrypt(self, m, key=None):
        key = key if key else self.pk
        return paillier.encrypt(m, *key)

    def _decrypt(self, c):
        return paillier.decrypt(c, *self.sk)

    # def singles(self, a=randint()):
    #     a_s = generate_shares(a)
    #     messages = []
    #     for i in range(n):
    #         keys = [(alpha, randint()) for alpha in self.alphas]
    #         macs = [calc_MAC(key, a_s[i]) for key in keys]
    #         messages.append([
    #             a_s[i],
    #             list(zip(keys, macs))
    #         ])
    #     return messages

    # def triples(self):
    #     a = randint()
    #     b = randint()
    #     c = (a * b) % p
    #     return [self.singles(a), self.singles(b), self.singles(c)]


if __name__ == "__main__":

    env = Env(n)
    try:
        # env.share(8)
        # env._forward()
        # env.mult_n(4)
        # env._forward()
        # env.add_macs(4)
        # env._clear()

        # env.singles(4)
        # env._clear()

        env.triples(4)
        env._publish()
        env._clear()

        # env.triples(4)
        # env._publish()
        # env._clear()
        # env.triples(6)
        # env._publish()
        # env._clear()

    finally:
        reg_print(env.parties)
        print(p)
        pass
