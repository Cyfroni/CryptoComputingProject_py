import random
import paillier
import re
import math
from time import sleep
from party import Party, parties_init


# paillier.encrypt = lambda m, *args: m
# paillier.decrypt = lambda c, *args: c

sbit = 16
# sbit = 4
primes = paillier.genPrimes(sbit)
# primes = (5, 7)

n = primes[0] * primes[1]
n2 = n * n


def reg_print(arg):
    limit = math.ceil(math.log10(n)) + 1
    reg = "\d{" + str(limit) + ",}"

    # arg = re.sub(reg, "**", str(arg))

    print(arg)


def randint():
    return random.randint(0, n - 1)


def randints(u):
    return [randint() for _ in range(u)]


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
    def addKeys(ak, bk):
        a_alpha, a_beta = ak
        b_alpha, b_beta = bk

        assert a_alpha == b_alpha

        return (a_alpha, (a_beta + b_beta) % n)

    @staticmethod
    def keys_mult_const(k, c):
        alpha, beta = k
        return (alpha, (beta * c) % n)

    @staticmethod
    def addition(a_s, b_s):
        a_, (aks, ams) = a_s
        b_, (bks, bms) = b_s

        r_ = (a_ + b_) % n

        rks = [
            Offline.addKeys(ak, bk) for ak, bk in zip(aks, bks)
        ]

        rms = [
            (am + bm) % n for am, bm in zip(ams, bms)
        ]

        return (r_, (rks, rms))

    @staticmethod
    def mult_const(a_s, c):
        a_, (aks, ams) = a_s

        r_ = (a_ * c) % n

        rks = [
            Offline.keys_mult_const(ak, c) for ak in aks
        ]

        rms = [
            (am * c) % n for am in ams
        ]

        return (r_, (rks, rms))

    @staticmethod
    def addition_const_1(a_s, c):
        a_, (aks, ams) = a_s

        r_ = (a_ + c) % n

        rks = aks[:]

        rms = ams[:]

        return (r_, (rks, rms))

    @staticmethod
    def addition_const_2(a_s, c):
        a_, (aks, ams) = a_s

        r_ = a_

        rks = aks[:]
        alpha, beta = rks[0]
        rks[0] = (alpha, (beta - c * alpha) % n)

        rms = ams[:]

        return (r_, (rks, rms))


def slow_pow(a, p):
    ret = 1
    for _ in range(p):
        ret = (ret * a) % n2

    return ret


class Env:

    def __init__(self, num_parties):
        self.parties = parties_init(num_parties, BDOZParty)
        sleep(1)

    def __forward(self):
        for party in self.parties:
            party.input = party.output

    def __clear(self):
        for party in self.parties:
            party.clear()

    def _share(self, u):
        for party in self.parties:
            xki = party.input_share if party.input_share else randints(u)
            Exki = [party._encrypt(x) for x in xki]
            party._broadcast(Exki)

            party.data["shares"] = (xki, Exki)

        sleep(0.1)

        for party in self.parties:
            data = party._receive_broadcast(party.data["shares"][1])

            party.output = list(zip(*data))

    def _mult_2(self, u, partyi, partyj):
        a = partyi.input_mult_2
        b = partyj.input_mult_2

        rs = randints(u)

        C = [
            (slow_pow(b[k], partyi._decrypt(a[k])) * partyi._encrypt(rs[k], partyj.pk)) % n2 for k in range(u)
        ]

        partyi._unicast(partyj.partyId, C)
        partyi.output = [-r % n for r in rs]

        sleep(0.1)

        vs = partyj._receive_unicast(partyi.partyId)

        partyj.output = [partyj._decrypt(v) % n for v in vs]

    def _mult_n(self, u):
        for party in self.parties:
            party.data['<ak>_'] = party.input[:u]
            party.data['<bk>_'] = party.input[u:]

            party.data['ak_'] = [
                ak[party.partyId] for ak in party.data['<ak>_']
            ]
            party.data['bk_'] = [
                bk[party.partyId] for bk in party.data['<bk>_']
            ]

            party.data['~ck_'] = [
                (party._decrypt(aki) * party._decrypt(bki)) %
                n for aki, bki in zip(party.data['ak_'], party.data['bk_'])
            ]

        for partyi in self.parties:
            for partyj in self.parties:
                if partyi.partyId == partyj.partyId:
                    continue

                partyi.input_mult_2 = partyi.data['ak_']
                partyj.input_mult_2 = partyj.data['bk_']

                self._mult_2(u, partyi, partyj)

                for party in [partyi, partyj]:
                    party.data['~ck_'] = [
                        (c + z) % n for c, z in zip(party.data['~ck_'], party.output)
                    ]

        for party in self.parties:
            party.input_share = party.data['~ck_']

        self._share(u)

    def _add_macs(self, u):
        for party in self.parties:
            party.data["#<ak>"] = party.input

            party.data["#ak"] = [
                ak[party.partyId] for ak in party.data['#<ak>']
            ]

            party.data["#alpha"] = randints(len(self.parties))
            party.data["#alpha"][party.partyId] = None

            party.data['#~[ak]'] = [
                (party._decrypt(ak), ([], [])) for ak in party.data["#ak"]
            ]

        for partyi in self.parties:
            for partyj in self.parties:
                if partyi.partyId == partyj.partyId:
                    continue
                alpha = partyi.data["#alpha"][partyj.partyId]

                alpha_enc = partyi._encrypt(alpha)

                partyi.input_mult_2 = [alpha_enc] * u
                partyj.input_mult_2 = partyj.data["#ak"]

                self._mult_2(u, partyi, partyj)

                betas = [-r % n for r in partyi.output]
                keys = [(alpha, beta) for beta in betas]
                macs = partyj.output

                for k in range(u):
                    partyi.data['#~[ak]'][k][1][0].append(keys[k])
                    partyj.data['#~[ak]'][k][1][1].append(macs[k])

        for party in self.parties:
            party.output = party.data['#~[ak]']

    def singles(self, u):
        if u <= 0:
            return

        self._share(u)

        self.__forward()

        self._add_macs(u)

        for party in self.parties:
            party.singles.extend(party.output[:])

        self.__clear()

    def triples(self, u):
        if u <= 0:
            return

        self._share(4 * u)

        for party in self.parties:
            party.data["_<ak>"] = party.output[:u]
            party.data["_<bk>"] = party.output[u:(2*u)]
            party.data["_<fk>"] = party.output[(2*u):(3*u)]
            party.data["_<gk>"] = party.output[(3*u):]

            party.input = party.data["_<ak>"] + party.data["_<bk>"]

        self._mult_n(u)

        for party in self.parties:
            party.data["_<ck>"] = party.output
            party.input = party.data["_<fk>"] + party.data["_<gk>"]

        self._mult_n(u)

        for party in self.parties:
            party.data["_<hk>"] = party.output

            party.input = party.data["_<ak>"] + party.data["_<bk>"] + party.data["_<ck>"] + \
                party.data["_<fk>"] + party.data["_<gk>"] + party.data["_<hk>"]

        self._add_macs(6 * u)

        for party in self.parties:
            party.data["_[ak]"] = party.output[:u]
            party.data["_[bk]"] = party.output[u:(2*u)]
            party.data["_[ck]"] = party.output[(2*u):(3*u)]
            party.data["_[fk]"] = party.output[(3*u):(4*u)]
            party.data["_[gk]"] = party.output[(4*u):(5*u)]
            party.data["_[hk]"] = party.output[(5*u):]

            party.triples.extend([
                (a, b, c) for a, b, c in zip(
                    party.data["_[ak]"],
                    party.data["_[bk]"],
                    party.data["_[ck]"]
                )
            ])

        # self._share(u)

        # for party in self.parties:
        #     party.data["_[ck]"] = party.output[:]

        #     party.input = party.data["_<fk>"]

        # for party in self.parties:
        #     party.data["_<ak>"] = party.output[:]

        # self._share(u)

        # for party in self.parties:
        #     party.data["_<bk>"] = party.output[:]

        # self._share(u)

        # for party in self.parties:
        #     party.data["_<fk>"] = party.output[:]

        # self._share(u)

        # for party in self.parties:
        #     party.data["_<gk>"] = party.output[:]

        #     party.input = party.data["_<ak>"] + party.data["_<bk>"]

        # self._mult_n(u)

        # for party in self.parties:
        #     party.data["_<ck>"] = party.output[:]
        #     party.input = party.data["_<fk>"] + party.data["_<gk>"]

        # self._mult_n(u)

        # for party in self.parties:
        #     party.data["_<hk>"] = party.output[:]

        #     party.input = party.data["_<ak>"]

        # self._add_macs(u)

        # for party in self.parties:
        #     party.data["_[ak]"] = party.output[:]

        #     party.input = party.data["_<bk>"]

        # self._add_macs(u)

        # for party in self.parties:
        #     party.data["_[bk]"] = party.output[:]

        #     party.input = party.data["_<ck>"]

        # self._add_macs(u)

        # for party in self.parties:
        #     party.data["_[ck]"] = party.output[:]

        #     party.input = party.data["_<fk>"]

        # self._add_macs(u)

        # for party in self.parties:
        #     party.data["_[fk]"] = party.output[:]

        #     party.input = party.data["_<gk>"]

        # self._add_macs(u)

        # for party in self.parties:
        #     party.data["_[gk]"] = party.output[:]

        #     party.input = party.data["_<hk>"]

        # self._add_macs(u)

        # for party in self.parties:
        #     party.data["_[hk]"] = party.output[:]

        #     # TODO: check

        #     party.triples.extend([
        #         (a, b, c) for a, b, c in zip(
        #             party.data["_[ak]"],
        #             party.data["_[bk]"],
        #             party.data["_[ck]"]
        #         )
        #     ])

        # self.__clear()

    ###

    def opening(self, varid, target_party):
        for party in self.parties:
            if target_party.partyId == party.partyId:
                continue

            a_s = party.vars[varid]
            a_, (_, ams) = a_s

            j = target_party.partyId
            if j > party.partyId:
                j -= 1

            m = (a_, ams[j])

            party._unicast(target_party.partyId, m)

        sleep(0.1)
        a_, (aks, _) = target_party.vars[varid]
        value = 0

        for party in self.parties:
            if target_party.partyId == party.partyId:
                value = (value + a_) % n
                continue
            v_, m = target_party._receive_unicast(party.partyId)

            j = party.partyId
            if j > target_party.partyId:
                j -= 1

            alpha, beta = aks[j]

            assert m == (alpha * v_ + beta) % n

            value = (value + v_) % n

        return value

    ###

    def initialize(self, s=10, t=10):
        self.singles(s)
        self.triples(t)

    def rand(self, varid):
        for party in self.parties:
            var = party.singles.pop()
            party.vars[varid] = var

    def add(self, varid1, varid2, varid3):
        for party in self.parties:

            x = party.vars[varid1]
            y = party.vars[varid2]

            party.vars[varid3] = Offline.addition(x, y)

    def multiply(self, varid1, varid2, varid3):
        for party in self.parties:

            x = party.vars[varid1]

            party.data["triple"] = party.triples.pop()
            a, _, _ = party.data["triple"]

            x_, _ = x
            a_, _ = a
            party.data["ep"] = (x_ - a_) % n
            party._broadcast([party.data["ep"]])

        sleep(0.1)

        for party in self.parties:

            ep = party._receive_broadcast([party.data["ep"]])
            party.data["ep"] = sum([x[0] for x in ep]) % n

        for party in self.parties:

            y = party.vars[varid2]

            _, b, _ = party.data["triple"]

            y_, _ = y
            b_, _ = b
            party.data["de"] = (y_ - b_) % n
            party._broadcast([party.data["de"]])

        sleep(0.1)

        for party in self.parties:

            ep = party.data["ep"]
            de = party._receive_broadcast([party.data["de"]])
            de = sum([x[0] for x in de]) % n

            addition_const = Offline.addition_const_1 if party.partyId == 0 else Offline.addition_const_2

            a, b, c = party.data["triple"]

            z1 = Offline.addition(
                c,
                Offline.mult_const(b, ep)
            )

            z2 = addition_const(
                Offline.mult_const(a, de),
                (ep * de) % n
            )

            party.vars[varid3] = Offline.addition(z1, z2)


class BDOZParty(Party):

    def __init__(self, *args):
        self.sk = paillier.keyGen(sbit, primes)  # n, g, lamb, miu
        self.pk = [self.sk[0], self.sk[1]]
        self.data = {}
        self.vars = {}
        self.singles = []
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
        singles({len(self.singles)}): {self.singles}
        triples({len(self.triples)}): {self.triples}

        input_share: {self.input_share}
        input_mult_2: {self.input_mult_2}
        input: {self.input}
        output: {self.output}

        vars({len(self.vars)}): {self.vars}
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
        # reg_print(f"{self.partyId}: broadcast {vals}")
        self.broadcast_message(self._to_message(vals))

    def _receive_broadcast(self, vals):
        messages = self._get_messages()
        # reg_print(f"{self.partyId}: received {messages}")

        ret = [0] * (len(messages) + 1)

        for party_id, message in messages.items():
            ret[int(party_id)] = self._to_vals(message)

        ret[self.partyId] = vals
        return ret

    def _unicast(self, target_id, vals):
        # reg_print(f"{self.partyId} -> {target_id}: unicast {vals}")
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


def open_with_assert(env, varid):
    var = env.opening(varid, env.parties[0])
    for party in env.parties:
        assert var == env.opening(varid, party)
    return var


def test_singles():
    num_parties = 3
    env = Env(num_parties)

    try:
        env.initialize(1, 0)

        env.rand("a")
        a = open_with_assert(env, "a")
        print(f"[{a}]")

    finally:
        sleep(3)
        reg_print(env.parties)


def test_triples():
    num_parties = 3
    env = Env(num_parties)

    try:
        env.initialize(0, 1)

        for party in env.parties:
            party.singles.extend(party.triples.pop())

        env.rand("c")
        env.rand("b")
        env.rand("a")

        a = open_with_assert(env, "a")
        b = open_with_assert(env, "b")
        c = open_with_assert(env, "c")

        # env.add("a", "b", "xx")

        print(f"[{a}] * [{b}] = [{c}] mod {n}")
        assert (a * b) % n == c

    finally:
        sleep(3)
        reg_print(env.parties)


def test_arit():
    num_parties = 3
    env = Env(num_parties)

    try:
        env.initialize(2, 1)
        env.rand("a")
        env.rand("b")

        a = open_with_assert(env, "a")
        b = open_with_assert(env, "b")

        env.add("a", "b", "c")
        c = open_with_assert(env, "c")
        print(f"[{a}] + [{b}] = [{c}] mod {n}")
        assert (a + b) % n == c

        env.multiply("a", "b", "d")
        d = open_with_assert(env, "d")
        print(f"[{a}] * [{b}] = [{d}] mod {n}")
        assert (a * b) % n == d

    finally:
        sleep(3)
        reg_print(env.parties)


def test():
    num_parties = 3
    env = Env(num_parties)

    def summ(env, sharid):
        _sum = 0
        for party in env.parties:
            _sum = (
                _sum + party._decrypt(party.data[sharid][0][party.partyId])) % n
        return _sum

    try:
        a, b, q, w, e, r = randints(6)
        env.parties[0].input_share = [q, w]
        env.parties[1].input_share = [e, r]
        env.parties[2].input_share = [
            (a-q-e) % n,
            (b-w-r) % n,
        ]

        env._share(2)
        for party in env.parties:
            party.input = party.output
        env._mult_n(1)
        for party in env.parties:
            party.data["<ck>"] = party.output

        ak = summ(env, "<ak>")
        bk = summ(env, "<bk>")
        ck = summ(env, "<ck>")
        print(f"<{ak}> * <{bk}> = <{ck}> mod {n}")
        assert ak == a
        assert bk == b
        assert ck == (a * b) % n

        for party in env.parties:
            party.input = party.data["<ak>"] + \
                party.data["<bk>"] + party.data["<ck>"]

        env._add_macs(3)

        for party in env.parties:
            party.singles.extend(party.output[:])

        env.singles(1)

        env.rand("xxx")
        env.rand("c")
        env.rand("b")
        env.rand("a")

        a = open_with_assert(env, "a")
        b = open_with_assert(env, "b")
        c = open_with_assert(env, "c")

        print(f"[{a}] * [{b}] = [{c}] mod {n}")
        assert (a * b) % n == c

        # print(summ(env, "<bk>"))
        # print(summ(env, "<ck>"))
        # print(env.parties[1]._decrypt(env.parties[1].data["<ak>"][0][1]))
        # print(env.parties[2]._decrypt(env.parties[2].data["<ak>"][0][2]))

        # env._add_macs(1)
        # for party in env.parties:
        #     party.singles = party.output
        # env.rand("a")
        # a = open_with_assert(env, "a")
        # print(a)
        # env._share(2)
        # for party in env.parties:
        #     party.input = party.output
        # env._mult_n(1)
    finally:
        reg_print(env.parties)


def test2():

    def summ(env, sharid):
        _sum = 0
        for party in env.parties:
            _sum = (
                _sum + party._decrypt(party.data[sharid][0][party.partyId])) % n
        return _sum

    num_parties = 3
    env = Env(num_parties)
    u = 1

    try:
        env._share(u)

        for party in env.parties:
            party.data["<ak>"] = party.output[:]

        env._share(u)

        for party in env.parties:
            party.data["<bk>"] = party.output[:]

            party.input = party.data["<ak>"] + party.data["<bk>"]

        env._mult_n(u)

        for party in env.parties:
            party.data["<ck>"] = party.output[:]

        ak = summ(env, "<ak>")
        bk = summ(env, "<bk>")
        ck = summ(env, "<ck>")
        print(f"<{ak}> * <{bk}> = <{ck}> mod {n}")
        assert ck == (ak * bk) % n

    finally:
        reg_print(env.parties)


if __name__ == "__main__":

    # test()
    # test2()
    # test_singles()
    # test_triples()
    test_arit()
    print(f"n: {n}")
