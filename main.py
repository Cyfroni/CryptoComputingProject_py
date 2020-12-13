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


if __name__ == "__main__":

    # test_singles()
    # test_triples()
    test_arit()
