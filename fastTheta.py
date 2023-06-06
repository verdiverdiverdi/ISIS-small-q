from math import floor
from numpy import zeros


class BoxedTheta:
    """
    For calculating the intersection of integer points within a box and ball
    """
    def __init__(self, R, q, coll_blocks=0):
        """
        Initalises BoxedTheta using three ``base case'' lists, which represent
        the possible bottoms of the recursion in __call__ below, and are addeed
        to cache

        :param R:           the radius of the ball
        :param q:           the side length of the box
        :param coll_blocks: the number of partitions to consider when using LSH
        """
        self.ell = floor(R**2+1)

        # initialise three lists for representing square lengths
        # Theta0 has all probability on length 0
        # Theta1 will give all square lengths greater than zero but less than
        # ell probability 2/q
        # Theta1_ will give the ``square length difference distribution''
        # within partitions when considering LSH blocks
        Theta0 = zeros(self.ell)
        Theta0[0] = 1
        Theta1 = zeros(self.ell)
        Theta1_ = zeros(self.ell)

        for i in range(q):
            x = min(i, q-i)
            try:
                Theta1[x**2] += 1./q
            except IndexError:
                # edge case where q large enough so that box contains ball
                pass

        c = 0
        f = coll_blocks
        for i in range(q):
            for j in range(q):
                # partitions q into f regions whenever f > 1
                if floor(f*i/q) != floor(f*j/q):
                    continue
                try:
                    Theta1_[(i-j)**2] += 1
                except IndexError:
                    # distance already too great, do not include in convolution
                    pass
                # count all pairs in normalisation constant, even if IndexError
                c += 1
        Theta1_ *= 1./c

        self.cache = dict({(0, 0): Theta0, (1, 0): Theta1, (0, 1): Theta1_})

    def convol(self, v, w):
        # Commutative. For speed, having v sparser than w is better
        x = v[0] * w
        for i in range(1, self.ell):
            if v[i] == 0:
                continue
            x[i:] += float(v[i]) * w[:-i]

        return x

    def __call__(self, n, ncoll=0):
        """
        Computes the probabilities of different square lengths at most self.ell
        after convolving the square length distributions of
            - n uniform (balanced) mod q
            - ncoll LSH distributions determined by coll_blocks in __init__

        :param n:       the number of uniform (balanced) mod q dimensions
        :param ncoll:   the number of LSH distribution dimensions

        :returns:       an array with the probability of square lengths less
                        than or equal to ell
        """

        if (n, ncoll) in self.cache:
            # if cached already, then return
            return self.cache[(n, ncoll)]

        if (n > 0):
            # consider the dimensions without LSH first, via recursion
            cached = self(n-1, ncoll)
            x = self.convol(self.cache[(1, 0)], cached)
            self.cache[(n, ncoll)] = x
            return x
        else:
            # then consider the LSH dimensions via recursion
            cached_ = self(0, ncoll-1)
            x = self.convol(self.cache[(0, 1)], cached_)
            self.cache[(0, ncoll)] = x
            return x
