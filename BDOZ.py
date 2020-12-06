import random
import paillier
import re
import math
from time import sleep
from party import Party, parties_init


# paillier.encrypt = lambda m, *args: m
# paillier.decrypt = lambda c, *args: c

# sbit = 256
sbit = 4
# primes = paillier.genPrimes(sbit)
primes = (5, 7)

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

    def _forward(self):
        for party in self.parties:
            party.input = party.output

    def _clear(self):
        for party in self.parties:
            party.clear()

    def _publish_triples(self):
        for party in self.parties:
            party.triples.extend(party.output)

    def _publish_singles(self):
        for party in self.parties:
            party.vars.extend(party.output)

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
            (slow_pow(b[k], partyi._decrypt(a[k])) * partyi._encrypt(rs[k], partyj.pk)) % n2 for k in range(u)
        ]

        partyi._unicast(partyj.partyId, C)
        partyi.output = [-r % n for r in rs]

        sleep(0.1)

        vs = partyj._receive_unicast(partyi.partyId)

        partyj.output = [partyj._decrypt(v) % n for v in vs]

    def mult_n(self, u):
        for party in self.parties:
            party.data['<ak>'] = party.input[: u]
            party.data['<bk>'] = party.input[u:]

            party.data['ak'] = [
                ak[party.partyId] for ak in party.data['<ak>']
            ]
            party.data['bk'] = [
                bk[party.partyId] for bk in party.data['<bk>']
            ]

            party.data['~ck'] = [
                (party._decrypt(aki) * party._decrypt(bki)) %
                n for aki, bki in zip(party.data['ak'], party.data['bk'])
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
                        (c + z) % n for c, z in zip(party.data['~ck'], party.output)
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

            party.data["alpha"] = randints(len(self.parties))
            party.data["alpha"][party.partyId] = None

            party.data['~[ak]'] = [
                (party._decrypt(ak), ([], [])) for ak in party.data["ak"]
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

                betas = [-r % n for r in partyi.output]
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

    def addition(self, varid1, varid2):
        for party in self.parties:

            a1 = party.vars[varid1]
            a2 = party.vars[varid2]

            party.vars.append(Offline.addition(a1, a2))

    def mult_const(self, varid, c):
        for party in self.parties:

            a = party.vars[varid]

            party.vars.append(Offline.mult_const(a, c))

    def addition_const(self, varid, c):
        for party in self.parties:

            a = party.vars[varid]

            if party.partyId == 0:
                party.vars.append(Offline.addition_const_1(a, c))
            else:
                party.vars.append(Offline.addition_const_2(a, c))


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


def test_triples():
    num_parties = 3
    env = Env(num_parties)

    try:
        env.triples(4)
        env._publish_triples()
        env._clear()
    except Exception as e:
        print(e)
    finally:
        reg_print(env.parties)


def open_with_assert(env, varid):
    var = env.opening(varid, env.parties[0])
    for party in env.parties:
        assert var == env.opening(varid, party)
    return var


def test_arit():
    num_parties = 3
    env = Env(num_parties)

    try:
        env.singles(2)
        env._publish_singles()
        env._clear()

        a = open_with_assert(env, 0)
        b = open_with_assert(env, 1)

        env.addition(0, 1)
        _r1 = open_with_assert(env, 2)
        assert (a + b) % n == _r1
        print(f"[{a}] + [{b}] = [{_r1}] mod {n}")

        r = randint()
        env.mult_const(0, r)
        _r2 = open_with_assert(env, 3)
        assert (a * r) % n == _r2
        print(f"[{a}] * {r} = [{_r2}] mod {n}")

        r = randint()
        env.addition_const(1, r)
        _r3 = open_with_assert(env, 4)
        assert (b + r) % n == _r3
        print(f"[{b}] + {r} = [{_r3}] mod {n}")

    except Exception as e:
        print(e)
    finally:
        reg_print(env.parties)


def test():
    num_parties = 3
    env = Env(num_parties)

    try:
        pass
    except Exception as e:
        print(e)
    finally:
        reg_print(env.parties)


if __name__ == "__main__":

    # test()
    # test_triples()
    test_arit()
    print(f"n: {n}")
