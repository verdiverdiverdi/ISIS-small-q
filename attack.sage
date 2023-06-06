from fpylll import BKZ, GSO, IntegerMatrix, LLL
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

from sage.all import IntegerModRing, ZZ, matrix, identity_matrix, \
        random_matrix, random_vector, vector
from g6k import Siever

from numpy import array, zeros
from numpy.linalg import norm

import sys

#  mitaka n512q257
n = 512
m_ = 1024
m = m_ + 1
q = 257
nu = 1470.71

Zq = IntegerModRing(q)

# ISIS matrix
Id = identity_matrix
A0 = random_matrix(Zq, m_-n, n)
# keeping the ``raw`` ISIS matrix to check everything at the end
AISIS = matrix.block(Zq, [[identity_matrix(ZZ, n)], [A0]])  # % q

# uniform ISIS target
u = random_vector(Zq, n)

# SIS* matrix
Au0 = matrix.block(Zq, [[matrix(u)], [A0]])  # % q
Afull = matrix.block(Zq, [[identity_matrix(ZZ, n)], [Au0]])  # % q
# keeping a numpy version of Afull makes life easier later
A_np = array(Afull.lift())

print("SIS* instance built")

# basis of the SIS* kernel lattice
B = matrix.block(ZZ, [[q*Id(n), 0], [-Au0, Id(m-n)]])
print("Full basis built")

"""
The basis B that has been built is the kernel for the SIS* matrix (columns)

A' = (Id || u ||  A0) in ZZ_q^(n x (n + 1 + n))
   = (Id || Au0)

where u is the uniform ISIS target vector and A0 is uniform in ZZ_q^(n x n).
We rewrite this as

A' = (A_1' A_2') = (Id || Au0)

The basis, in ~row~ notation, is

[q I_n     0   ]
[  U   I_(n+1) ]

where U = - (A_1')^-1 A_2' = -Au0
"""

# take a 2z dimensional block, with volume q^z
# i.e. symmetric around the centre of the basis
z1 = 80
z2 = 80
d = z1+z2
"""
Note that because of the form of the basis, the basis of the
projected sublattice we are concerned with can be directly
taken from the full basis as below.

First we project against some number k of q vectors to take

[q I_n     0   ]
[  U   I_(n+1) ]

to

[0        0         0   ]
[0  q I_(n - k)     0   ]
[0    U[k:n]    I_(n+1) ]

where U[k:n] is the kth to the (n-1)th columns of U.
Then we simply do not include the final k row vectors.
"""
B_ = matrix(B)[n-z1:n+z2, n-z1:n+z2]

beta_BKZ = 12
beta_sieve = 60


def complete_solution(v):
    # this lifts and reduces a vector from the projected sublattice sieve
    x = zeros(m, dtype="int32")
    x[n-z1:n+z2] = v
    y = (x.dot(A_np))[:n-z1] % q
    y -= q * (y > q/2)
    x[:n-z1] = -y
    return x


print("Partial basis built")
C = B_.LLL()
print("Partial Basis LLL reduced")

X = IntegerMatrix.from_matrix(C)
M = GSO.Mat(X, float_type="ld", U=IntegerMatrix.identity(d),
            UinvT=IntegerMatrix.identity(d))
lll = LLL.Reduction(M)
bkz = BKZ2(lll)
g6k = Siever(M)

for bs in range(5, beta_BKZ+1):
    param = BKZ.Param(block_size=bs, max_loops=1, auto_abort=True)
    print("\rBKZ-%d / %d ... " % (bs, beta_BKZ), end="")
    bkz(param)
    bkz.lll_obj()
    sys.stdout.flush()
print("BKZ profile :")

for x in bkz.M.r():
    print("%.3f" % (x**.5), end=" ")
print

g6k.initialize_local(0, beta_sieve-20, beta_sieve)
g6k(alg="gauss")
while g6k.l > 0:
    g6k.extend_left()
    g6k(alg="gauss" if g6k.n < 50 else "hk3")
    print("\r Sieve-%d / %d ... " % (g6k.n, beta_sieve), end="")
    sys.stdout.flush()

with g6k.temp_params(saturation_ratio=.9):
    g6k(alg="gauss" if g6k.n < 50 else "hk3")


print("\n Sieving Done")

norms = []

X_ = array(matrix(X))[:beta_sieve]
print(X_.shape)

trials = 0
FailZ, FailC, FailN = 0, 0, 0
for vec in g6k.itervalues():
    trials += 1
    v = array(vec)
    x = v.dot(X_)

    if (x % q == 0).all():
        # trivial vector: all mod q
        FailZ += 1
        continue

    if abs(x[z1]) != 1:
        # we do not recieve +/-1 in the first position, we we cannot
        # solve ISIS for u
        FailC += 1
        continue

    lx = norm(array(x))
    y = complete_solution(x)
    ly = norm(y)
    if ly < nu:
        break
    # the norm of the solution is too long
    FailN += 1


print("Failures: \n\t %d lifts were 0 mod q,\n\t %d lifts didn't had the coeff +/- 1,\n\t %d lifts were too long" % (FailZ, FailC, FailN)) # noqa
if trials == g6k.db_size():
    print("FAILED: All candidates lifted. No solution found")
    exit()

# Reconstructing ISIS solution from SIS* solution
f = - y[n]
x = vector(ZZ, list(f * y[:n])+list(f * y[n+1:]))

# Checking it all
assert (x * AISIS == u)
assert (x.norm().n() < nu)

# Claiming victory
print("SUCCESS: ISIS solved after %d lifts, out of %d candidates !" % (trials, g6k.db_size())) # noqa
print("Solution Norm:", x.norm().n(), " < ", nu)
print("solution :", x)
