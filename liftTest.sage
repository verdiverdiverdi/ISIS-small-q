from zShape import genSIS

from fpylll import BKZ, GSO, IntegerMatrix, LLL
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

from copy import deepcopy
from multiprocessing import Pool

from sage.all import ceil, randint, set_random_seed, ZZ, matrix

try:
    from g6k import Siever, SieverParams
except ModuleNotFoundError:
    pass

import os
import pickle


def loops(bs):
    if bs <= 5:
        return 8
    if bs <= 10:
        return 4
    if bs <= 20:
        return 2
    return 1


def floats(m):
    if m <= 200:
        return "ld"
    if m <= 250:
        return "dd"
    return "qd"


class liftExperiment:
    def __init__(self, q, n, m):
        """
        Initialise the experiment with parameters

        :param q:   the modulus of the SIS instance
        :param n:   the SIS kernel lattice has volume q^n
        :param m:   the SIS kernel lattice has rank m
        """
        self.q = q
        self.n = n
        self.m = m
        self.f = floats(m)

    def generate(self):
        """
        Returns a SIS kernel lattice basis
        """
        return genSIS(self.q, self.n, self.m)

    def findqs(self):
        """
        After BKZ reduction find how large the block of q vectors remaining at
        the beginning of the basis is

        :returns:   a non negative integer determining the size of the topleft
                        block of q vectors
        """
        self.bkz.M.discover_all_rows()
        self.bkz.M.update_gso()
        profile = self.bkz.M.r()
        bools = [all([profile[:i] == tuple([self.q**2]*i) for i in range(j+2)])
                 for j in range(self.m)]
        qBlockLen = bools.index(False)
        return qBlockLen

    def qVectorsCombo(self, vecqReduced):
        """
        A vector which consists of only combinations of the first n qvectors
        is trivial

        :param vecqReduced: a vector with its first left entries reduced mod q
        :returns:       a bool that if True denotes a trivial solution
        """

        first = all([vecqReduced[i] % self.q == 0 for i in range(self.n)])
        second = all([vecqReduced[i] == 0 for i in range(self.n, self.m)])
        return first and second

    def findBestLift(self, g6kSiever, left, right):
        """
        Having sieved a SIS kernel lattice basis in the projected sublattice
        defined by B_[l:r], and assuming B[0], ..., B[l - 1] are qvectors,
        lift the sieve database over the projected lattice to the full lattice

        We ignore lifted vectors that are sums of qvectors

        :param g6kSiever:   a g6k Siever object holding a sieve database
        :param left:        the beginning index of the projected lattice
        :param right:       the end index (exclusive) of the projected lattice

        """
        # vec * restrictedB is an unreduced lift that will ultimately hav
        # its first ``left`` coordinates reduced mod q
        restrictedB = g6kSiever.M.B[left:right]

        # creating the projected lattice basis where the sieve occurred
        A = matrix(ZZ, right-left, g6kSiever.full_n)
        restrictedB.to_matrix(A)
        for i in range(A.nrows()):
            for j in range(left):
                A[i, j] = 0

        # vec * projectedB is a vector in the projected sieve database
        projectedB = IntegerMatrix.from_matrix(A)

        def qReduce(vec, left):
            """
            Takes a lattice vector ``vec`` and reduces the first ``left`` many
            coordinates mod q around 0
            """
            vec = list(vec)
            for i in range(left):
                qMod = vec[i] % self.q
                vec[i] = min(qMod, self.q - qMod)
            return tuple(vec)

        # projSqrnrms counts the number of vectors in the projected sieve
        # database of a given square length
        projSqrnrms = {}
        # qSqrnrms counts the number of lifted vectors with their first
        # ``left`` vectors reduced mod q of a given square length
        qSqrnrms = {}
        # leftqSqrnrms counts the same as qSqrnrms, but restricted to only
        # the first ``left`` entries
        leftqSqrnrms = {}

        totalLong = 0
        totalTriv = 0

        upperBound = ceil(4./3 * self.q**2)

        for vec in g6kSiever.itervalues():
            projLatticeVec = projectedB.multiply_left(vec)
            projSqrnrm = sum([projLatticeVec[i]**2 for i in range(self.m)])

            if projSqrnrm > upperBound:
                totalLong += 1
                continue

            projSqrnrms[projSqrnrm] = projSqrnrms.get(projSqrnrm, 0) + 1

            latticeVec = restrictedB.multiply_left(vec)
            # lift and reduce the first ``left`` positions
            vecqReduced = qReduce(latticeVec, left)

            # check not "trivial"
            if self.qVectorsCombo(vecqReduced):
                totalTriv += 1
                continue

            # leftvecSqrnrm is the norm of the lifted and reduced mod q part
            leftqSqrnrm = sum([vecqReduced[i]**2 for i in range(left)])
            leftqSqrnrms[leftqSqrnrm] = leftqSqrnrms.get(leftqSqrnrm, 0) + 1

            # vecSqrnrm is the full norm of the lifted vector
            qSqrnrm = leftqSqrnrm + sum([vecqReduced[i]**2 for i in range(left, self.m)])

            qSqrnrms[qSqrnrm] = qSqrnrms.get(qSqrnrm, 0) + 1

        return projSqrnrms, qSqrnrms, leftqSqrnrms

    def reduce(self, max_bs, liftStart):
        """
        Reduce a SIS kernel lattice up to blocksize ``max_bs``


        :param max_bs:      the maximum blocksize to perform reduction
        :param liftStart:   the blocksizes at which to attempt to find a
                                solution via lifting

        :returns:           a dictionary of results
        """
        M = GSO.Mat(self.Bfpl, float_type=self.f)
        lll = LLL.Reduction(M)
        self.bkz = BKZ2(lll)
        self.bkz.lll_obj()

        res = {}

        for bs in range(3, max_bs+1):
            projSqrnrms, qSqrnrms, leftqSqrnrms, lifted, left = self.bsLift(bs, lift=(bs in liftStart))
            profile = self.bkz.M.r()
            res[bs] = {"projSqrnrms": projSqrnrms, "qSqrnrms": qSqrnrms,
                       "leftqSqrnrms": leftqSqrnrms, "profile": profile,
                       "lifted": lifted, "left": left}

        return res

    def bsLift(self, bs, lift=False):
        """
        Perform BKZ reduction and if ``lift`` is ``True`` then attempt to
        find a solution via lifting provided a q block still exists at the
        start of the basis

        :param bs:      a blocksize for reduction and for the sieve context
        :param lift:    a bool, if ``True`` attempt the lifting attack

        :returns:       dictionaries qSqrnrms and leftqSqrnrms as described
                            above, and a bool and non negative integer
                            lifted: if ``True`` lifting occurred
                            left: the number of q vectors remaining

        """
        param = BKZ.Param(block_size=bs,
                          max_loops=loops(bs),
                          auto_abort=True)
        self.bkz(param, 0, self.m)
        self.bkz.lll_obj()

        sqrnrm = self.bkz.M.r()[0]

        if not lift:
            lifted = False
            left = None
            projSqrnrms = {sqrnrm: 1}
            qSqrnrms = {sqrnrm: 1}
            leftqSqrnrms = {sqrnrm: 1}
            return projSqrnrms, qSqrnrms, leftqSqrnrms, lifted, left
        else:
            left = self.findqs()
            right = min(self.m, left+bs)

            if left == 0:
                # still possible that sieving in [0, right) will solve, but
                # this is not what we are checking for
                lifted = False
                projSqrnrms = {sqrnrm: 1}
                qSqrnrms = {sqrnrm: 1}
                leftqSqrnrms = {sqrnrm: 1}
                return projSqrnrms, qSqrnrms, leftqSqrnrms, lifted, left

            lifted = True

            Bg6k = deepcopy(self.bkz.M.B)
            Mg6k = GSO.Mat(Bg6k, float_type=self.f,
                           U=IntegerMatrix.identity(self.m),
                           UinvT=IntegerMatrix.identity(self.m))

            g6kparams = SieverParams(otf_lift=False,
                                     saturation_ratio=.99)
            g6kSiever = Siever(Mg6k, g6kparams)
            g6kSiever.initialize_local(left, left, right)
            g6kSiever(alg="gauss")
            originalSize = g6kSiever.db_size()
            databaseSize = ceil((4./3)**(.5*(right-left)))
            g6kSiever.resize_db(databaseSize)
            if originalSize < databaseSize:
                g6kSiever(alg="gauss")

            # lifting over q vectors
            projSqrnrms, qSqrnrms, leftqSqrnrms = self.findBestLift(g6kSiever,
                                                                    left,
                                                                    right)

            return projSqrnrms, qSqrnrms, leftqSqrnrms, lifted, left

    def __call__(self, max_bs, liftStart=None):
        """
        Solve an instance as defined by __init__

        :param max_bs:      the maximum blocksize to reach progressively
        :param liftStart:   the blocksizes at which to lift

        :returns:           res: a dictionary with keys equal to blocksizes and
                            values = (sqrnrm, lifted, left), a squared length
                            of the shortest vector found, and whether this was
                            found via lifting or not, and the length of the
                            qblock
        """
        if liftStart is None:
            liftStart = [i for i in range(3, max_bs+1)]

        self.Bfpl = self.generate()
        res = self.reduce(max_bs, liftStart=liftStart)

        for bs in liftStart:
            # sort dictionaries by square norms
            res[bs]['projSqrnrms'] = dict(sorted(res[bs]['projSqrnrms'].items()))
            res[bs]['qSqrnrms'] = dict(sorted(res[bs]['qSqrnrms'].items()))
            res[bs]['leftqSqrnrms'] = dict(sorted(res[bs]['leftqSqrnrms'].items()))

        return res


def oneExp(params):
    (q, n, m, max_bs, liftStart, seed) = params
    set_random_seed(seed + randint(0, 2**31))
    lE = liftExperiment(q, n, m)
    return lE(max_bs, liftStart=liftStart)


def manyExps(exps, procs, q, n, m, max_bs, liftStart=None, save=False,
             addendum=None):

    if liftStart is None:
        liftStart = [max_bs]

    jobs = [(q, n, m, max_bs, liftStart, seed) for seed in range(exps)]
    with Pool(procs) as p:
        res = p.map(oneExp, jobs)
    res = [x for x in res if x is not None]

    if save:
        # making sure data/ exists
        if not os.path.exists("data"):
            os.makedirs("data")

        filename = "lifts-{q}-{n}-{m}-{bs}-{lift}-{exp}".format(q=q, n=n, m=m,
                                                                bs=max_bs,
                                                                lift=liftStart,
                                                                exp=exps)
        if addendum is not None:
            filename += "-" + addendum

        filename += ".pkl"

        with open("data/" + filename, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return res
