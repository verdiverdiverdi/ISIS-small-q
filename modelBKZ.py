from math import pi, exp, log, sqrt, floor
# import functools

# (very) slightly adapted from
# https://github.com/pq-crystals/security-estimates/blob/master/model_BKZ.py


def delta_0f(k):
    """
    Auxiliary function giving root Hermite factors. Small values
    experimentally determined, otherwise from [Chen13]
    :param k: BKZ blocksize for which the root Hermite factor is required
    """
    small = (( 2, 1.02190),  # noqa
             ( 5, 1.01862),  # noqa
             (10, 1.01616),
             (15, 1.01485),
             (20, 1.01420),
             (25, 1.01342),
             (28, 1.01331),
             (40, 1.01295))

    k = float(k)
    if k <= 2:
        return (1.0219)
    elif k < 40:
        for i in range(1, len(small)):
            if small[i][0] > k:
                return (small[i-1][1])
    elif k == 40:
        return (small[-1][1])
    else:
        return (k/(2*pi*exp(1)) * (pi*k)**(1./k))**(1/(2*(k-1.)))


def svp_classical(b):
    """ log_2 of best known classical cost of SVP in dimension b
    """
    return b * log(sqrt(3./2))/log(2)    # .292 * b [BeckerDucasLaarhovenGama]


# Adding Memoisation to this slow function
# @functools.lru_cache(maxsize=2**20)
def construct_BKZ_shape(q, nq, n1, b):
    """
    Simulates the (log) shape of a basis after the reduction of a
    [q ... q, 1 ... 1] shape by BKZ-b reduction (nq many q, n1 many 1)
    This is implemented by constructing a longer shape and looking
    for the subshape with the right volume.
    We also output the index of the first vector <q and the first =1.

    :param q:   a modulus
    :param nq:  the number of q-vectors beginning the basis
    :param n1:  the number of 1-vectors (after projection) ending the basis
    :param b:   a BKZ blocksize

    :returns:   a triple (a, a + B, L) where:
        - a is the number of q-vectors remaining at the beginning of the basis,
            or alternatively the index of the first (projected) vectors shorter
            than q
        - a + B is the number of q-vectors and the number of (projected)
            vectors in the sloped region of the basis, or alternatively the
            index of the first (projected) 1-vector
        - L is the log shape of the entire basis

    ..note::    this implentation takes O(n). It is possible to output a
                compressed description of the shape in time O(1), but it is
                more error prone and requires a case analysis
    """
    d = nq+n1
    lq = log(q)
    glv = nq*lq  # Goal log volume

    slope = -2*log(delta_0f(b))

    if b == 0:
        L = nq*[log(q)] + n1*[0]
        return (nq, nq, L)

    B = int(floor(lq / -slope))  # number of vectors in the sloped region
    L = nq*[lq] + [lq + i * slope for i in range(1, B+1)] + n1*[0]

    x = 0
    lv = sum(L[:d])

    # while current volume exceeeds goal volume, slide window to the right
    while lv > glv:
        lv -= L[x]
        lv += L[x+d]
        x += 1

    assert x <= B  # Sanity check that we have not gone too far

    L = L[x:x+d]
    a = max(0, nq - x)             # The length of the [q, ... q] sequence
    B = min(B, d - a)              # The length of the GSA sequence

    # Sanity check the volume, up to the discreteness of index error
    diff = glv - lv
    assert abs(diff) < lq

    # Small shift of the GSA sequence to equilibrate volume
    for i in range(a, a+B):
        L[i] += diff / B
    lv = sum(L)

    assert abs(lv/glv - 1) < 1e-6        # Sanity check the volume

    # checking we have the boundaries of q-vectors and sloped region correct
    assert L[a] < log(q)
    if a != 0:
        assert L[a-1] == log(q)
    assert L[a + B - 1] > 0
    if a + B < d:
        assert L[a + B] == 0

    return (a, a + B, L)


def construct_BKZ_shape_randomised(q, nq, n1, b):
    """
    Simulates the (log) shape of a [q ... q, 1 ... 1] shape after randomisation
    and then BKZ-b reduction (such that no GS vectors gets smaller than 1)

    :param q:   a modulus
    :param nq:  the number of q-vectors beginning the basis
    :param n1:  the number of 1-vectors (after projection) ending the basis
    :param b:   a BKZ blocksize

    :returns:   a triple (a, a + B, L) where:
        - a is the number of q-vectors remaining at the beginning of the basis,
            or alternatively the index of the first (projected) vectors shorter
            than q
        - a + B is the number of q-vectors and the number of (projected)
            vectors in the sloped region of the basis, or alternatively the
            index of the first (projected) 1-vector
        - L is the log shape of the entire basi

    ..note::    retaining the 1 vectors at the end of the basis is equivalent
                to implicitly choosing the correct m = nq + n1 to minimise the
                cost of the attack, hence this method may give a shorter first
                SIS kernel lattice basis length than rhfFirstLen above
    """
    glv = nq * log(q)
    d = nq+n1
    L = []

    slope = -2 * log(delta_0f(b))
    li = 0
    lv = 0

    # sum the slope section until larger than goal log volume
    for i in range(d):
        li -= slope
        lv += li
        if lv > glv:
            break
        L = [li]+L

    # The length of the sloped sequence
    B = len(L)
    L += (d-B)*[0]
    # The length of the [q, ... q] sequence
    a = 0

    lv = sum(L)
    diff = lv - glv
    # Sanity check the volume, up to the discretness of index error
    # assert abs(diff) < li
    # Small shift of the GSA sequence to equilibrate volume
    for i in range(a, a+B):
        L[i] -= diff / B
    lv = sum(L)
    assert abs(lv/glv - 1) < 1e-6        # Sanity check the volume

    return (a, a + B, L)


def BKZ_first_length(q, nq, n1, b):
    """ Simulates the length of the shortest expected vector in the first
    b-block after randomisation (killing q-vectors) and a BKZ-b reduction.

    :param q:   a modulus
    :param nq:  the number of q-vectors beginning the basis
    :param n1:  the number of 1-vectors (after projection) ending the basis
    :param b:   a BKZ blocksize

    :returns:   an estimate for the first length of a SIS kernel lattice basis
    """

    (_, _, L) = construct_BKZ_shape_randomised(q, nq, n1, b)
    firstl = exp(L[0])  # Compute the root-volume of the first block
    return firstl
