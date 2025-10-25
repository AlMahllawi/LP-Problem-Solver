from fractions import Fraction

def to_subscript(n):
    """Converts an integer to a string of Unicode subscript characters."""
    subs = "₀₁₂₃₄₅₆₇₈₉"
    return "".join([subs[int(d)] for d in str(n)])

def format_number(num):
    """Formats a number as an integer or a fraction string."""
    if isinstance(num, int) or (isinstance(num, float) and num.is_integer()):
        return str(int(num))

    # For floats that are very close to an integer
    if abs(num - round(num)) < 1e-9:
        return str(round(num))

    f = Fraction(num).limit_denominator(1000)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"