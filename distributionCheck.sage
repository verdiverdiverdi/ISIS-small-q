from math import ceil, floor, sqrt

from fastTheta import BoxedTheta
from small_qSIS import successProb

from sage.all import list_plot, log

import os
import pickle


def checkLeftLiftDist(data, q, bs, verbose=False):
    """
    A function for checking the distribution of experimental lifts against the
    uniform distribution in the cube

    :param data:    a list entry of a .pkl file saved from ``manyExps`` in
                        liftTest
    :param q:       the q used to generate data
    :param bs:      the bs used to generate data
    """
    leftqSqrnrms = data[bs]['leftqSqrnrms']
    qBlockLen = data[bs]['left']

    # assuming lifts fall uniformly
    averageSqrnrm = qBlockLen * q**2. / 12.

    datapoints = sum(leftqSqrnrms.values())

    avgTotal = 0
    for (sqrnrm, number) in leftqSqrnrms.items():
        avgTotal += (sqrnrm * number)
    expAverage = float(avgTotal / datapoints)

    if verbose:
        print(averageSqrnrm, expAverage)

    # more finegrained, what fraction of our leftqSqrnrms do we expect to be
    # below a certain length (sqrnrm) is given by modelCDF, and the what we
    # observe experimentally is given by expCDF
    soFar = 0
    results = {}

    for (sqrnrm, number) in leftqSqrnrms.items():
        soFar += number
        if verbose:
            print(soFar)
        expCDF = float(soFar) / datapoints
        R = sqrt(sqrnrm)
        bt = BoxedTheta(R, q)
        modelCDF = sum(bt(qBlockLen))
        if verbose:
            print(expCDF, modelCDF, expCDF / modelCDF)
            print()
        results[(sqrnrm, number)] = (soFar, expCDF, modelCDF)

    return averageSqrnrm, expAverage, results


def checkFullLiftDist(data, q, bs, verbose=False):
    """
    A function for checking the distribution of the lengths of full lifts
    against our model

    :param data:    a list entry of a .pkl file saved from ``manyExps`` in
                        liftTest
    :param q:       the q used to generate data
    :param bs:      the bs used to generate data
    """
    qSqrnrms = data[bs]['qSqrnrms']
    projSqrnrms = data[bs]['projSqrnrms']
    qBlockLen = data[bs]['left']

    assert qBlockLen > 0, 'no q vectors!'

    soFar = 0
    results = {}

    for (qSqrnrm, qNumber) in qSqrnrms.items():
        soFar += qNumber
        if verbose:
            print(soFar)
        modelExpectedNumber = 0
        # given projNumber of projected square norms of size projSqrnrm, how
        # many of our qSqrnrms do we expect to be below qSqrnrm
        # according to our model this is ``modelExpectedNumber``
        # according to experimental observations it is ``soFar``
        for (projSqrnrm, projNumber) in projSqrnrms.items():
            log2Success = min(0,
                              successProb(q, qBlockLen, ceil(sqrt(qSqrnrm)),
                                          floor(sqrt(projSqrnrm))) / log(2.)
                              )
            projSuccess = 2**log2Success
            modelExpectedNumber += (projNumber * projSuccess)

        if verbose:
            print(soFar, modelExpectedNumber, soFar / modelExpectedNumber)
        results[(qSqrnrm, qNumber)] = (soFar, modelExpectedNumber)

    return results


def makeLiftPlot(data, q, bs, verbose=False, save=False, addendum=None,
                 plot=False):
    """
    A function for saving and plotting our lifting model against experimental
    data

    :param data:    a list entry of a .pkl file saved from ``manyExps`` in
                        liftTest
    :param q:       the q used to generate data
    :param bs:      the bs used to generate data
    :param save:    if ``True`` save the modelled and experimental values of
                        the proportions of lifts we expect to have a particular
                        length
    :param plot:    if ``True`` print plots of what is described in save above

    ..note::    ``checkFullLiftDist`` is terribly optimised in terms of memory
                and the cache ``bt_cache`` that is used to store convolutions
                of theta series grows very large... For me this runs on a
                machine with 64GiB of RAM

    ..note::    the plots output here are not the ones used in the paper, which
                are generated in the .tex
    """

    _, _, leftLift = checkLeftLiftDist(data, q, bs, verbose=verbose)
    leftLiftPointsExp = []
    leftLiftPointsMod = []
    for ((sqrnrm, _), (_, expCDF, modelCDF)) in leftLift.items():
        leftLiftPointsExp += [(round(sqrt(sqrnrm)), expCDF)]
        leftLiftPointsMod += [(round(sqrt(sqrnrm)), modelCDF)]

    if plot:
        pltLeft = list_plot(leftLiftPointsExp, axes_labels=['$L_i\ \ $  ', "\n $p'(L_i, q, n_q)$  "]) # noqa
        pltLeft += list_plot(leftLiftPointsMod, color='red')
        if verbose:
            pltLeft.show()

    fullLift = checkFullLiftDist(data, q, bs, verbose=verbose)
    fullLiftPointsExp = []
    fullLiftPointsMod = []
    for ((qSqrnrm, _), (soFar, modelExpectedNumber)) in fullLift.items():
        fullLiftPointsExp += [(round(sqrt(qSqrnrm)), soFar)]
        fullLiftPointsMod += [(round(sqrt(qSqrnrm)), modelExpectedNumber)]

    if plot:
        pltFull = list_plot(fullLiftPointsExp, axes_labels=['$L_{total, k}\ \ \ \ \ \ \ $         ', 'expected']) # noqa
        pltFull += list_plot(fullLiftPointsMod, color='red')
        if verbose:
            pltFull.show()

    if save:
        if not os.path.exists("data"):
            os.makedirs("data")

        filenameLeft = "{q}-{bs}-left".format(q=q, bs=bs)
        filenameFull = "{q}-{bs}-full".format(q=q, bs=bs)

        if addendum is not None:
            filenameLeft += "-" + addendum
            filenameFull += "-" + addendum

        with open('data/' + filenameLeft + '.pkl', 'wb') as handle:
            pickle.dump((leftLiftPointsExp, leftLiftPointsMod), handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        with open('data/' + filenameFull + '.pkl', 'wb') as handle:
            pickle.dump((fullLiftPointsExp, fullLiftPointsMod), handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        if plot:
            pltLeft.save("data/" + filenameLeft + ".png")
            pltFull.save("data/" + filenameFull + ".png")
