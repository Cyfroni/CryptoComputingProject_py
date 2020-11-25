import paillier
import random

num_players = 5
modulus = 97
alpha = 10


def assert_equal(a, b):
    assert a % modulus == b % modulus


def generate_shares(value):
    r = [random.randint(0, modulus - 1) for _ in range(num_players - 1)]
    vals = [(value - sum(r)) % modulus] + r

    assert_equal(sum(vals), value)

    return vals


def share_val(val, alpha_i):
    x_i = [val] + [0] * (num_players - 1)
    mac_i = [a * val for a in alpha_i]
    val_i = list(zip(x_i, mac_i))

    assert_equal(val, open_val_i(val_i))

    return val_i


def open_val_i(val_i):
    x_i, mac_i = zip(*val_i)
    val = sum(x_i) % modulus

    assert_equal(val * alpha,  sum(mac_i))

    return val


def add_val_i(a_i, b_i):
    c_i = []
    for a_, b_ in zip(a_i, b_i):
        xa_, maca_ = a_
        xb_, macb_ = b_
        c_i.append((xa_ + xb_, maca_ + macb_))

    assert_equal(open_val_i(a_i) + open_val_i(b_i), open_val_i(c_i))

    return c_i


# def mult_val_i(x_i, y_i, triple):
#     z_i = []
#     a_i, b_i, c_i = triple

#     epsilon = p_open_vals_i(x_i, a_i)
#     rho = p_open_vals_i(y_i, b_i)

#     # [z] = [c] + ep * [b] + rho * [a] + ep * rho

#     return z_i


def p_open_vals_i(a_i, b_i):
    return (sum(a_i[0]) - sum(b_i[0])) % modulus


# init
players = [paillier.keyGen(256) for _ in range(num_players)]
alpha_i = generate_shares(alpha)

# share a = 5
a = 5
a_i = share_val(a, alpha_i)


# share b = 8
b = 8
b_i = share_val(b, alpha_i)

# compute c = a + b
c_i = add_val_i(a_i, b_i)

# compute d = a * b
# d_i = mult_val_i(a_i, b_i)

# share triples [a], [b], [c] such c = b * c


# enc_shares = [paillier.encrypt(s, n, g)
#               for (n, g, lamb, miu), s in zip(players, shares)]

# print(enc_shares)
