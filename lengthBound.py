from math import log, pi, sqrt

"""
This file computes the length bound of the SIS^* or ISIS instances implied by
[C:ETWY22].

The constants are taken from p.16 of https://eprint.iacr.org/2022/785.pdf
"""
alpha_F = 1.17
alpha_M512 = 2.04
alpha_M1024 = 2.33
tau = 1.04

leps_F = 36
leps_M = 41


def smoothing(d, leps):
    # there is an error on p.16 of https://eprint.iacr.org/2022/785.pdf
    # both sigma_Falcon and sigma_Mitaka have a dependence on d and should
    # be larger than the smoothing parameter on ZZ^2d with relevant epsilon
    return (1./pi) * sqrt(log(4*d*(1+2**leps)) / 2)


def length(d, q, falcon=True):
    if falcon:
        return tau * smoothing(d, leps_F) * alpha_F * sqrt(q*2*d)
    else:
        # Mitaka
        if d == 512:
            return tau * smoothing(d, leps_M) * alpha_M512 * sqrt(q*2*d)
        elif d == 1024:
            return tau * smoothing(d, leps_M) * alpha_M1024 * sqrt(q*2*d)


for d in [512, 1024]:
    for q in [257, 521, 1031]:
        for falcon in [True, False]:
            if falcon:
                scheme = "falcon"
            else:
                scheme = "mitaka"
            print(d, q, scheme)
            print(length(d, q, falcon=falcon))
            print()
