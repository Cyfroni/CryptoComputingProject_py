import random
import re
import math
from time import sleep

from .networking import BDOZParty, parties_connect
from .encryption import paillier


def randint(n):
    return random.randint(0, n - 1)


def randints(u, n):
    return [randint(n) for _ in range(u)]


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
    def keys_add(ak, bk, n):
        a_alpha, a_beta = ak
        b_alpha, b_beta = bk

        assert a_alpha == b_alpha

        return (a_alpha, (a_beta + b_beta) % n)

    @staticmethod
    def keys_mult_const(k, c, n):
        alpha, beta = k
        return (alpha, (beta * c) % n)

    @staticmethod
    def addition(a_s, b_s, n):
        a_, (aks, ams) = a_s
        b_, (bks, bms) = b_s

        r_ = (a_ + b_) % n

        rks = [
            Offline.keys_add(ak, bk, n) for ak, bk in zip(aks, bks)
        ]

        rms = [
            (am + bm) % n for am, bm in zip(ams, bms)
        ]

        return (r_, (rks, rms))

    @staticmethod
    def mult_const(a_s, c, n):
        a_, (aks, ams) = a_s

        r_ = (a_ * c) % n

        rks = [
            Offline.keys_mult_const(ak, c, n) for ak in aks
        ]

        rms = [
            (am * c) % n for am in ams
        ]

        return (r_, (rks, rms))

    @staticmethod
    def addition_const_1(a_s, c, n):
        a_, (aks, ams) = a_s

        r_ = (a_ + c) % n

        rks = aks[:]

        rms = ams[:]

        return (r_, (rks, rms))

    @staticmethod
    def addition_const_2(a_s, c, n):
        a_, (aks, ams) = a_s

        r_ = a_

        rks = aks[:]
        alpha, beta = rks[0]
        rks[0] = (alpha, (beta - c * alpha) % n)

        rms = ams[:]

        return (r_, (rks, rms))


def slow_pow(a, p, n2):
    ret = 1
    for _ in range(p):
        ret = (ret * a) % n2

    return ret


class BDOZ(BDOZParty):

    def __init__(self, sk):
        n, g, _, _ = sk
        self.sk = sk
        self.pk = (n, g)
        self.singles = []
        self.triples = []
        self.vars = {}

        self.input_share = None
        self.input_mult_2 = None
        self.input = None
        self.output = []
        self.local = {}

        self.data = {}

    def __repr__(self):
        return f"""
        ## {self.partyId} ##
        singles({len(self.singles)}): {self.singles}
        triples({len(self.triples)}): {self.triples}
        vars({len(self.vars)}): {self.vars}
        
        data: {self.data}
        """

    def full_clear(self):
        self.vars = []
        self.triples = []
        self.output = []
        self.clear()

    def protocol_clear(self):
        self.input_share = None
        self.input_mult_2 = None
        self.input = None
        self.output = []
        self.local = {}

    def _encrypt(self, m, key=None):
        key = key if key else self.pk
        return paillier.encrypt(m, *key)

    def _decrypt(self, c):
        return paillier.decrypt(c, *self.sk)


class Env:

    def __init__(self, num_parties, sbit):

        primes = paillier.genPrimes(sbit)

        self.n = primes[0] * primes[1]
        self.n2 = self.n * self.n

        self.parties = [BDOZ(paillier.keyGen(sbit, primes))
                        for _ in range(num_parties)]
        parties_connect(self.parties)
        sleep(1)

    def _forward(self):
        for party in self.parties:
            party.input = party.output

    def _protocol_clear(self):
        for party in self.parties:
            party.protocol_clear()

    def _share(self, u):
        for party in self.parties:
            xki = party.input_share if party.input_share else randints(
                u, self.n)
            Exki = [party._encrypt(x) for x in xki]
            party._broadcast(Exki)

            party.local["Exki"] = Exki

        sleep(0.1)

        for party in self.parties:
            data = party._receive_broadcast(party.local["Exki"])

            party.output = list(zip(*data))

    def _mult_2(self, u, partyi, partyj):
        a = partyi.input_mult_2
        b = partyj.input_mult_2

        rs = randints(u, self.n)

        C = [
            (slow_pow(b[k], partyi._decrypt(a[k]), self.n2) * partyi._encrypt(rs[k], partyj.pk)) % self.n2 for k in range(u)
        ]

        partyi._unicast(partyj.partyId, C)
        partyi.output = [-r % self.n for r in rs]

        sleep(0.1)

        vs = partyj._receive_unicast(partyi.partyId)

        partyj.output = [partyj._decrypt(v) % self.n for v in vs]

    def _mult_n(self, u):
        for party in self.parties:
            party.local['*/<ak>'] = party.input[:u]
            party.local['*/<bk>'] = party.input[u:]

            party.local['*/ak'] = [
                ak[party.partyId] for ak in party.local['*/<ak>']
            ]
            party.local['*/bk'] = [
                bk[party.partyId] for bk in party.local['*/<bk>']
            ]

            party.local['*/~ck'] = [
                (party._decrypt(aki) * party._decrypt(bki)) % self.n
                for aki, bki in zip(party.local['*/ak'], party.local['*/bk'])
            ]

        for partyi in self.parties:
            for partyj in self.parties:
                if partyi.partyId == partyj.partyId:
                    continue

                partyi.input_mult_2 = partyi.local['*/ak']
                partyj.input_mult_2 = partyj.local['*/bk']

                self._mult_2(u, partyi, partyj)

                for party in [partyi, partyj]:
                    party.local['*/~ck'] = [
                        (c + z) % self.n for c, z in zip(party.local['*/~ck'], party.output)
                    ]

        for party in self.parties:
            party.input_share = party.local['*/~ck']

        self._share(u)

    def _add_macs(self, u):
        for party in self.parties:
            party.local["M/<ak>"] = party.input

            party.local["M/ak"] = [
                ak[party.partyId] for ak in party.local['M/<ak>']
            ]

            party.local["M/alpha"] = randints(len(self.parties), self.n)
            party.local["M/alpha"][party.partyId] = None

            party.local['M/~[ak]'] = [
                (party._decrypt(ak), ([], [])) for ak in party.local["M/ak"]
            ]

        for partyi in self.parties:
            for partyj in self.parties:
                if partyi.partyId == partyj.partyId:
                    continue
                alpha = partyi.local["M/alpha"][partyj.partyId]

                alpha_enc = partyi._encrypt(alpha)

                partyi.input_mult_2 = [alpha_enc] * u
                partyj.input_mult_2 = partyj.local["M/ak"]

                self._mult_2(u, partyi, partyj)

                betas = [-r % self.n for r in partyi.output]
                keys = [(alpha, beta) for beta in betas]
                macs = partyj.output

                for k in range(u):
                    partyi.local['M/~[ak]'][k][1][0].append(keys[k])
                    partyj.local['M/~[ak]'][k][1][1].append(macs[k])

        for party in self.parties:
            party.output = party.local['M/~[ak]']

    def singles(self, u):
        if u <= 0:
            return

        self._share(u)

        self._forward()

        self._add_macs(u)

        for party in self.parties:
            party.singles.extend(party.output[:])

        self._protocol_clear()

    def triples(self, u):
        if u <= 0:
            return

        self._share(4 * u)

        for party in self.parties:
            party.local["t/<ak>"] = party.output[:u]
            party.local["t/<bk>"] = party.output[u:(2*u)]
            party.local["t/<fk>"] = party.output[(2*u):(3*u)]
            party.local["t/<gk>"] = party.output[(3*u):]

            party.input = party.local["t/<ak>"] + party.local["t/<bk>"]

        self._mult_n(u)

        for party in self.parties:
            party.local["t/<ck>"] = party.output
            party.input = party.local["t/<fk>"] + party.local["t/<gk>"]

        self._mult_n(u)

        for party in self.parties:
            party.local["t/<hk>"] = party.output

            party.input = party.local["t/<ak>"] + party.local["t/<bk>"] + party.local["t/<ck>"] + \
                party.local["t/<fk>"] + \
                party.local["t/<gk>"] + party.local["t/<hk>"]

        self._add_macs(6 * u)

        for party in self.parties:
            party.local["t/[ak]"] = party.output[:u]
            party.local["t/[bk]"] = party.output[u:(2*u)]
            party.local["t/[ck]"] = party.output[(2*u):(3*u)]
            party.local["t/[fk]"] = party.output[(3*u):(4*u)]
            party.local["t/[gk]"] = party.output[(4*u):(5*u)]
            party.local["t/[hk]"] = party.output[(5*u):]

            party.triples.extend([
                (a, b, c) for a, b, c in zip(
                    party.local["t/[ak]"],
                    party.local["t/[bk]"],
                    party.local["t/[ck]"]
                )
            ])

        self._protocol_clear()

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
                value = (value + a_) % self.n
                continue
            v_, m = target_party._receive_unicast(party.partyId)

            j = party.partyId
            if j > target_party.partyId:
                j -= 1

            alpha, beta = aks[j]

            assert m == (alpha * v_ + beta) % self.n

            value = (value + v_) % self.n

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

            party.vars[varid3] = Offline.addition(x, y, self.n)

    def multiply(self, varid1, varid2, varid3):
        for party in self.parties:

            x = party.vars[varid1]

            party.local["m/triple"] = party.triples.pop()
            a, _, _ = party.local["m/triple"]

            x_, _ = x
            a_, _ = a
            party.local["m/ep"] = (x_ - a_) % self.n
            party._broadcast([party.local["m/ep"]])

        sleep(0.1)

        for party in self.parties:

            ep = party._receive_broadcast([party.local["m/ep"]])
            party.local["m/ep"] = sum([x[0] for x in ep]) % self.n

        for party in self.parties:

            y = party.vars[varid2]

            _, b, _ = party.local["m/triple"]

            y_, _ = y
            b_, _ = b
            party.local["m/de"] = (y_ - b_) % self.n
            party._broadcast([party.local["m/de"]])

        sleep(0.1)

        for party in self.parties:

            ep = party.local["m/ep"]
            de = party._receive_broadcast([party.local["m/de"]])
            de = sum([x[0] for x in de]) % self.n

            addition_const = Offline.addition_const_1 if party.partyId == 0 else Offline.addition_const_2

            a, b, c = party.local["m/triple"]

            z1 = Offline.addition(
                c,
                Offline.mult_const(b, ep, self.n),
                self.n
            )

            z2 = addition_const(
                Offline.mult_const(a, de, self.n),
                (ep * de) % self.n,
                self.n
            )

            party.vars[varid3] = Offline.addition(z1, z2, self.n)

        self._protocol_clear()
