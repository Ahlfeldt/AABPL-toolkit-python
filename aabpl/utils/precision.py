from math import (
    sin as _math_sin, cos as _math_cos, atan2 as _math_atan2, pi as _math_pi,
    acos as _math_acos, log10 as _math_log10,
    factorial as _math_factorial
)
from decimal import Decimal as _decimal_Decimal, getcontext as _decimal_getcontext

def sin_taylor(x, decimals):
    p = 0
    _decimal_getcontext().prec = decimals
    for n in range(decimals):
        p += _decimal_Decimal(((-1)**n)*(x**(2*n+1)))/(_decimal_Decimal(_math_factorial(2*n+1)))
    return p


def cos_taylor(x, decimals):
    p = 0
    _decimal_getcontext().prec = decimals
    for n in range(decimals):
        p += _decimal_Decimal(((-1)**n)*(x**(2*n)))/(_decimal_Decimal(_math_factorial(2*n)))
    return p
