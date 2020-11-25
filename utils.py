def extended_gcd(a, b):
    """The extended Euclidean algorithm."""
    x = 0
    lastx = 1
    y = 1
    lasty = 0
    while b != 0:
        quotient = a // b
        a, b = b, a % b
        x, lastx = lastx - quotient*x, x
        y, lasty = lasty - quotient*y, y
    return (lastx, lasty, a)
