from src import Env
from time import sleep


def open_with_assert(env, varid):
    var = env.opening(varid, env.parties[0])
    for party in env.parties:
        assert var == env.opening(varid, party)
    return var


def test_singles():
    num_parties = 3
    env = Env(num_parties, 16)

    try:
        env.initialize(1, 0)

        env.rand("a")
        a = open_with_assert(env, "a")
        print(f"[{a}]")

    finally:
        sleep(3)
        print(env.parties)
        print(env.n)


def test_triples():
    num_parties = 3
    env = Env(num_parties, 16)

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

        print(f"[{a}] * [{b}] = [{c}] mod {env.n}")
        assert (a * b) % env.n == c

    finally:
        sleep(3)
        print(env.parties)
        print(env.n)


def test_arit():
    num_parties = 3
    env = Env(num_parties, 16)

    try:
        env.initialize(2, 1)
        env.rand("a")
        env.rand("b")

        a = open_with_assert(env, "a")
        b = open_with_assert(env, "b")

        env.add("a", "b", "c")
        c = open_with_assert(env, "c")
        print(f"[{a}] + [{b}] = [{c}] mod {env.n}")
        assert (a + b) % env.n == c

        env.multiply("a", "b", "d")
        d = open_with_assert(env, "d")
        print(f"[{a}] * [{b}] = [{d}] mod {env.n}")
        assert (a * b) % env.n == d

    finally:
        sleep(3)
        print(env.parties)
        print(env.n)


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
        print(env.parties)


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
        print(env.parties)


if __name__ == "__main__":

    # test()
    # test2()
    # test_singles()
    # test_triples()
    test_arit()
