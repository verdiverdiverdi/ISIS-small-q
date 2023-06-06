# create (and possibly save) the values for Table 2
# the values from left to right are the (rounded up values) of
#   SISs
#   ISIS_specific
#   SISs_lift
#   ISIS_lift_specific

from lengthBound import length
from small_qSIS import SIS_optimize_attack

import os
import pickle

ds = [512]
qs = [257, 521, 1031]
schemes = ["falcon", "mitaka"]


def run(q, m, n, nu, scheme, save=False):
    """
    Run our estimates for particular parameters for each attack in

        {no otf lifting, otf lifting} x {SIS^* x Lemma 1 reduction x q/2 loss}

    in particular the label ``inhom`` when = "worst" assumes a probability loss
    factor of approximately mq and when "specific" assumes a probability
    loss factor of q/2
    """
    print(scheme, q, m, n, nu)
    nu = int(nu)
    costs = {}
    costs["SIS_SVP"] = SIS_optimize_attack(q, m, n, nu, sieve="svp_only")
    costs["SISs"] = SIS_optimize_attack(q, m, n, nu, otf_lift=False)
    costs["ISIS_worst"] = SIS_optimize_attack(q, m, n, nu, otf_lift=False,
                                              inhom="worst")
    costs["ISIS_specific"] = SIS_optimize_attack(q, m, n, nu, otf_lift=False,
                                                 inhom="specific")
    costs["SISs_lift"] = SIS_optimize_attack(q, m, n, nu, otf_lift=True)
    costs["ISIS_lift_worst"] = SIS_optimize_attack(q, m, n, nu, otf_lift=True,
                                                   inhom="worst")
    costs["ISIS_lift_specific"] = SIS_optimize_attack(q, m, n, nu,
                                                      otf_lift=True,
                                                      inhom="specific")
    print()

    if save:
        if not os.path.exists("data"):
            os.makedirs("data")
        filename = "{q}-{m}-{n}-{nu}-".format(q=q, m=m, n=n, nu=nu)
        filename += (scheme + ".pkl")
        with open('data/' + filename, 'wb') as handle:
            pickle.dump(costs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def runAll(save=False):
    for d in ds:
        for q in qs:
            for scheme in schemes:
                if scheme == "falcon":
                    falcon = True
                elif scheme == "mitaka":
                    falcon = False
                nu = length(d, q, falcon=falcon)

                run(q, 2*d, d, nu, scheme, save=save)
