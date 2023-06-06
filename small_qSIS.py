# the main engine for estimating the cost of attacks
# we were unable to achieve any interesting gains using LSH techniques, but
# we leave the skeleton of the idea here and in fastTheta.py

from math import log, sqrt, floor
from modelBKZ import svp_classical, BKZ_first_length, construct_BKZ_shape
from fastTheta import BoxedTheta

log_infinity = 9999
STEPS_b = 1
bt_cache = {}


def logIntersectionProportion(n, R, q, ncoll=0):
    """
    Consider the ball of radius R and the centred box of side length q.
    Return the probability that a uniform vector mod q (possibly with some
    coordinates following a different distribution implied by locality
    sensitive hashing) falls into the ball.

    :param n:       the number of uniform coordinates mod q
    :param R:       the radius of the ball
    :param q:       the side length of the cube
    :param ncoll:   the number of coordinates determined by LSH techniques

    :returns:       the (natural) log of the described probability
    """
    if (R, q) in bt_cache.keys():
        bt = bt_cache[(R, q)]
    else:
        bt = BoxedTheta(R, q)
        bt_cache[(R, q)] = bt
    proportion = bt(n, ncoll=ncoll)
    return log(sum(proportion))


def SIS_l2_cost(q, w, h, nu, b, cost_svp=svp_classical, verbose=False,
                **kwargs):
    """
    Returns the cost of finding a vector shorter than nu with BKZ-b.
    The equation is Ax = 0 mod q, where A has h rows, and w columns
    (h equations in dim w).

    Here we do randomise the basis to remove the q vector structure.

    :param q:           the modulus
    :param w:           the rank of the SIS kernel lattice is w
    :param h:           the volume of the SIS kernel lattice is q^h
    :param nu:          the ell_2 length bound for the SIS solution
    :param b:           the BKZ blocksize
    :param cost_svp:    the log2 cost of BKZ as a function of b
    """

    firstLen = BKZ_first_length(q, h, w-h, b)
    if firstLen > nu:
        return log_infinity
    if verbose:
        print("Attack uses block-size %d and %d equations" % (b, h))
        print("shortest vector used has length l=%.2f, q=%d, l<q = %s" % (
            firstLen, q, firstLen < q))
    return cost_svp(b)


def successProb(q, qBlockLen, nu, projLen):
    """
    Return the probability that the lift of single projected vector of length
    projLen is shorter than nu, i.e. the probability of a single lift solving
    SIS*

    :param q:           the modulus
    :param qBlockLen:   the number of qvectors to lift over
    :param nu:          the ell_2 length bound for the SIS solution
    :param projLen:     the length of the projected vector

    :returns:           the (natural) log of the estimated probability or
                        -log_infinity if suspected unsolvable via this method
    """
    assert qBlockLen > 0, 'no q vectors!'

    if projLen > nu:
        return -log_infinity

    R = sqrt(nu**2 - projLen**2)
    logSuccess = logIntersectionProportion(qBlockLen, R, q, ncoll=0)
    return logSuccess


def SIS_l2_cost_qvec(q, w, h, nu, b, cost_svp=svp_classical, verbose=False,
                     sieve="bdgl", otf_lift=True, collision=False,
                     inhom=None):
    """
    Returns the cost of finding a vector shorter than nu with BKZ-b if it
    works. The equation is Ax = 0 mod q, where A has h rows, and w columns
    (h equations in dim w).

    Here we do ~not~ randomise the basis and therefore keep q vector structure.

    :param q:           the modulus
    :param w:           the rank of the SIS kernel lattice is w
    :param h:           the volume of the SIS kernel lattice is q^h
    :param nu:          the ell_2 length bound for the SIS solution
    :param b:           the BKZ blocksize
    :param cost_svp:    the log2 cost of BKZ as a function of b
    :param sieve:       the sieve design used, when otf is used affects the
                            number of vectors lifted and their length
    :param otf_lift:    a bool that if False only lifts vectors in the terminal
                            sieving database, else considers lifting vectors
                            seen during sieving
    :param collision:   a bool that if True does some light locality sensitive
                            hashing on lifted vectors (experimental)
    :param inhom:       whether we solve the SIS^* problem or ISIS, and if ISIS
                            which probability loss factor we use
                                "worst": the generic ~mq from Lemma 1, or
                                "specific": the q/2 our SIS^* solver achieves
    """

    assert (sieve in {"svp_only", "nv", "bdgl"})
    assert (inhom is None or inhom in {"worst", "specific"})

    if inhom in {"worst", "specific"}:
        w += 1

    (qBlockLen, _, _) = construct_BKZ_shape(q, h, w-h, b)
    firstLen = q

    if qBlockLen == 0 or sieve == "svp_only":
        # BKZ-b already removed the q vector structure
        return SIS_l2_cost(q, w, h, nu, b, cost_svp=cost_svp, verbose=verbose)

    if not otf_lift:
        # Saturated ball of radius sqrt(4/3)
        projLen = firstLen * sqrt(4/3)
        logProjs = b * log(4./3)/2.
    else:
        # Unsaturated ball of larger radius
        if sieve == "nv":
            projLen = firstLen * sqrt(4/3) * sqrt(2)
            logProjs = b * log(4./3)
        if sieve == "bdgl":
            # bdgl visits fewer vectors than nv in a sieve iteration
            # but those that are seen are shorter
            projLen = firstLen * sqrt(2)
            logProjs = b * log(3./2)/2.

    if projLen > nu:
        return log_infinity

    if collision:
        # choose to collide on the maximum number of dimensions that does not
        # increase our memory above that of sieving
        # ASSUMPTION: we get at least one vector per bucket having made this
        # choice
        # ASSUMPTION: we can put a vector of minimum sqr length (4/3 q) in each
        # of these buckets despite potentially using them during otf_lifting
        # EXPERIMENTAL
        nn_coll = floor(min(log(4./3)/(2*log(2)) * b, qBlockLen))
        R = sqrt(nu**2 - projLen**2 - (4/3) * firstLen**2)
        nn_nocoll = qBlockLen - nn_coll
        logSuccess = logIntersectionProportion(nn_nocoll, R, q, ncoll=nn_coll)
        log_p = min(0, logSuccess + logProjs)
        cost = cost_svp(b) - log_p/log(2)
    else:
        logSuccess = successProb(q, qBlockLen, nu, projLen)
        # in all three cases below we assume the success probability
        # matches the union bound
        if inhom is None:
            # solve SIS*
            log_p = min(0, logSuccess + logProjs)
        if inhom == "worst":
            # solve ISIS with reduction for arbitrary adversaries
            log_p = min(0, logSuccess + logProjs - log(w*(q-1)))
            # solve ISIS using our SIS* solver as the adversary
        if inhom == "specific":
            log_p = min(0, logSuccess + logProjs + log(2./q))
        cost = cost_svp(b) - log_p/log(2.)

    if verbose:
        print("BKZ blocksize \t=", b)
        print("lift dim \t=", qBlockLen)
        if collision:
            print("coll dim \t=", nn_coll)
        print("R_qlift \t=", round(float(sqrt(nu**2 - projLen**2)), 5))
        print("log2 Success \t=", round(float(logSuccess / log(2.)), 5))
        print("log2_#targets \t=", round(float(logProjs / log(2)), 5))
        print()

        print("log2_p =", round(float(log_p / log(2)), 5))
        print("cost =", round(float(cost), 5))

    return cost


def SIS_optimize_attack(q, w, h, nu, cost_attack=SIS_l2_cost_qvec,
                        cost_svp=svp_classical, verbose=True, sieve="bdgl",
                        otf_lift=True, collision=False, inhom=None):
    """
    Find optimal parameters for a given attack

    :param q:           the modulus
    :param w:           the rank of the SIS kernel lattice is w
    :param h:           the volume of the SIS kernel lattice is q^h
    :param nu:          the ell_2 length bound for the SIS solution
    :param cost_attack: the attack style (removing or abusing q vectors)
    :param cost_svp:    the log2 cost of BKZ as a function of b
    :param sieve:       the sieve design used, affects the number of vectors
                            in the terminating sieve database, and their length
    :param otf_lift:    a bool that if False only lifts vectors in the terminal
                            sieving database, else considers lifting vectors
                            seen during sieving
    :param collision:   a bool that if True does some light locality sensitive
                            hashing on lifted vectors
    :param inhom:       whether we solve the SIS^* problem or ISIS, and if ISIS
                            which probability loss factor we use
                                "worst": the generic mq from Lemma 1, or
                                "specific": the q/2 our SIS^* solver achieves
    """
    best_cost = log_infinity

    for b in range(w, 2, - STEPS_b):
        if verbose:
            print("\r b = %d \t" % b, end="")

        cost = cost_attack(q, w, h, nu, b, cost_svp, verbose=False,
                           sieve=sieve, otf_lift=otf_lift, collision=collision,
                           inhom=inhom)
        if cost <= best_cost:
            best_cost = cost
            if verbose:
                print("cost = %.3f" % cost, end="")
            best_b = b

        if cost >= best_cost + 10:
            # assuming we have then found the global minimum
            break

    if verbose:
        print("\r" + 40*" ")

    if verbose:
        cost_attack(q, w, h, nu, best_b, cost_svp=cost_svp, verbose=verbose,
                    sieve=sieve, otf_lift=otf_lift, collision=collision,
                    inhom=inhom)

    return (best_b, best_cost)
