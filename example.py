from small_qSIS import SIS_optimize_attack

"""
An example against the q = 257, n = 512, Falcon parameters suggested in
[C:ETWY22].

We first try a method equivalent to randomising the basis and performing BKZ
reduction (Generic Attack)

Then, without on the fly lifting, we perform
    the SIS^* attack, and
    the ISIS attack with probability loss factor q/2 described in Sec 3.5

Then, with on the fly lifting considering the Becker--Ducas--Gama--Laarhoven
sieve, we perform the same
"""

print("\n == Generic Attack ==")
SIS_optimize_attack(257, 1024, 512, 801, sieve="svp_only")
print("\n == Lift Sieve Database Attack ==")
SIS_optimize_attack(257, 1024, 512, 801, otf_lift=False)
SIS_optimize_attack(257, 1024, 512, 801, otf_lift=False, inhom="specific")
print("\n == On The Fly Lift Sieve Attack ==")
SIS_optimize_attack(257, 1024, 512, 801, otf_lift=True)
SIS_optimize_attack(257, 1024, 512, 801, otf_lift=True, inhom="specific")
