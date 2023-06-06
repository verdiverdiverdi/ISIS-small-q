# if you alter this run ./refresh.sh

from collections import OrderedDict

from fpylll import BKZ, GSO, IntegerMatrix, LLL
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

from modelBKZ import construct_BKZ_shape

from multiprocessing import Pool

from sage.all import matrix, identity_matrix, zero_matrix, ZZ, randint, \
        set_random_seed, ceil, sqrt, log, list_plot

import os
import pickle


def genSIS(q, n, m):
    """
    Generate a SIS kernel lattice basis in row form
        [ q I_n   0    ]
        [   U    I_m-n ]
    where U is uniformly random in {0, ..., q-1} and transform into an fpylll
    integer matrix

    :param q:   the modulus of the SIS instance
    :param n:   the SIS kernel lattice has volume q^n
    :param m:   the SIS kernel lattice has rank m
    :returns:   fpylll integer matrix of above form
    """
    A00 = q * identity_matrix(ZZ, n)
    A11 = identity_matrix(ZZ, m-n)
    A10 = matrix(ZZ, [[randint(0, q-1) for _ in range(n)] for _ in range(m-n)])
    A01 = zero_matrix(ZZ, n, m-n)
    A = matrix.block(ZZ, [[A00, A01], [A10, A11]])
    B = IntegerMatrix.from_matrix(A)
    return B


def reduce(B, max_bs, profiles=True, randomise=False, right=None):
    """
    Reduce a basis B using fpylll up to blocksize max_bs, possibly randomising
    before reduction (to remove q e_i vectors from the beginning) and possibly
    returning the basis profiles after each blocksize of BKZ reduction

    :param B:           an fpylll matrix
    :param max_bs:      the maximum blocksize for BKZ reduction
    :param profiles:    a bool, if ``True`` return a dictionary of profiles
                            after each blocksize of lattice reduction
    :param randomise:   if bool, if ``True`` remove the q e_i vectors from the
                            beginning of SIS kernel lattice basis
    :param right:       if ``None`` perform lattice reduction on the entire
                            basis, else perform lattice reduction up to index
                            right
    :returns:           if profiles is ``True`` a dictionary of profiles, else
                            the fpylll GSO object after the BKZ reduction

    ..note::            we use ``right`` to attempt to leave one vectors at the
                            end of a reduced basis in our Zshape
                            experiments
    """
    assert B.nrows == B.ncols
    m = B.nrows

    if right is None:
        right = m

    if profiles:
        profs = OrderedDict()

    float_type = "ld"
    if m >= 200:
        float_type = "dd"
    if m >= 250:
        float_type = "qd"

    M = GSO.Mat(B, float_type=float_type)

    lll = LLL.Reduction(M)
    bkz = BKZ2(lll)

    if randomise:
        bkz.lll_obj()
        bkz.randomize_block(0, m, density=ceil(m/2))

    bkz.lll_obj(kappa_start=0, kappa_end=right)
    if profiles:
        profs[2] = bkz.M.r()

    for bs in range(3, max_bs+1):
        # rough attempt to make sure the progressive BKZ is sufficient to
        # follow the heuristics used in our modelBKZ for our Z shape without
        # becoming too costly
        if bs >= 3:
            loops = 8

        param = BKZ.Param(block_size=bs, max_loops=loops, auto_abort=True)
        bkz(param, 0, right)
        bkz.lll_obj(kappa_start=0, kappa_end=right)
        if profiles:
            profs[bs] = bkz.M.r()

    if profiles:
        return profs
    else:
        return M


def oneExp(params):
    q, n, m, max_bs, ones, seed = params
    set_random_seed(randint(0, 2**31) + seed)
    B = genSIS(q, n, m)
    if ones:
        right = construct_BKZ_shape(q, n, m-n, max_bs)[1]
    else:
        right = None
    profiles = reduce(B, max_bs, right=right)
    return profiles


def manyExps(exps, procs, q, n, m, max_bs, save=True, addendum=None,
             ones=True):
    jobs = [(q, n, m, max_bs, ones, seed) for seed in range(exps)]
    with Pool(procs) as p:
        res = p.map(oneExp, jobs)
    res = [x for x in res if x is not None]

    if save:
        # checking data/ exists
        if not os.path.exists("data"):
            os.makedirs("data")

        filename = "z-{q}-{n}-{m}-{bs}-{exp}".format(q=q, n=n, m=m, bs=max_bs,
                                                     exp=exps)
        if addendum is not None:
            filename += "-" + addendum

        filename += ".pkl"

        with open("data/" + filename, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return res


def plotZ(data, n, m, q, max_bs, save=False, plot=False, stats=True):
    """
    A function for saving and plotting the output of Zshape experiments

    :param data:    a .pkl file saved from ``manyExps`` in zShape
    :param n:       the n used to generate data
    :param m:       the m used to generate data
    :param q:       the q used to generate data
    :param max_bs:  the max_bs used to generate data
    :param save:    if ``True`` save the average profile created from data and
                        the profile expected by our model as .txt files
    :param plot:    if ``True`` print plots of what is described in save above
    :param stats:   if ``True`` print the sample average and standard deviation
                        of the number of q vectors for Table I

    ..note::    the plots output here are not what are included in the .pdf,
                those are created in the .tex source using the saved .txt files
    """
    relevantBS = [5, 10, 20, 30, 40, 50]
    trials = len(data)
    bs = [i for i in relevantBS if i <= max_bs]
    for b in bs:
        numqs = [0]*trials
        avgs = [0]*m
        L = construct_BKZ_shape(q, n, m-n, b)[2]
        for trial in range(trials):
            numqs[trial] = data[trial][b].count(q**2)
            for i in range(m):
                avgs[i] += (sqrt(data[trial][b][i]) / trials)
        avgs = [log(x) for x in avgs]

        if stats:
            Eqs = sum(numqs) / trials
            Vqs = sum([x**2 for x in numqs]) / 60 - Eqs**2
            print(b, float(Eqs), float(sqrt(Vqs)))

        if save:
            if not os.path.exists("data"):
                os.makedirs("data")

            fname = "{n}-{m}-{q}-{b}-average".format(q=q, n=n, m=m, b=b)
            with open('data/' + fname + '.txt', 'w') as f:
                for i in range(m):
                    f.write(str(i) + ',' + str(avgs[i]))
                    f.write('\n')

            fnameModel = "{n}-{m}-{q}-{b}-model".format(q=q, n=n, m=m, b=b)
            with open('data/' + fnameModel + '.txt', 'w') as f:
                for i in range(m):
                    f.write(str(i) + ',' + str(L[i]))
                    f.write('\n')

        if plot:
            plt = list_plot(L)
            plt += list_plot(avgs, color='red')
            plt.show()
