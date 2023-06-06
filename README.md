In this folder
- attack.sage: a custom attack script for the smallest q Mitaka parameters
- distributionCheck.sage: functions that take in data from liftTest.sage and save and plot various figures comparing our model to experimental data
- example.py: a simple example of how to run our main attack cost script against some parameters
- fastTheta.py: our method for convolving theta series to determine the number of integer points in balls and boxes
- lengthBound.py: a helper script that extracts nu from the various parameters suggested in [C:ETWY22]
- liftTest.sage: a script that runs our attack and collects data about the lengths of projected vectors in the sieve database and of lifted vectors
- modelBKZ.py: functions that model the profiles output by BKZ reduction on q-ary lattice bases, initially from [Dilithium](https://github.com/pq-crystals/security-estimates/blob/master/model_BKZ.py).
- runParams.py: run our attack cost script against small q Falcon and Mitaka instances
- small_qSIS.py: our main cost estimation script, can cost generic BKZ attacks that apply randomisation to remove the q vectors as well as our attack
- zShape.sage: a script that runs experiments regarding the Z-shape of q-ary lattice bases after reduction

Note that refresh.sh simply creates a python friendly version of zShape.sage for other files to import functions from it

In data/
- pickles of the form 1031-1024-512-1606-falcon.pkl contain the attack cost estimates as a dictionary for these parameters given by runParams.py
- text files of the form 120-240-257-5-XXXX.txt contain either the average experimental or modelled basis profile for these parameters from zShape.sage or modelBKZ.py respectively
- pickles of the form 257-40-XXXX.pkl contain the data extracted by distributionCheck.sage from experiments run in liftTest.sage, and one can see plots in the respective .png
- text files of the form experimentalXXXXLift.txt and modelXXXXLift.txt are .tex friendly versions of the data 257-40-XXXX.pkl for plotting
- the pickle lifts-257-120-240-40-[40]-10.pkl is the raw data from liftTest.sage
- the pickle z-257-120-240-40-60-ones-heavy.pkl is the raw data from zShape.sage

NOTE:
    You will need the general sieve kernel installed in sage.
    This can be achieved via installing sage onto your system via a method https://doc.sagemath.org/html/en/installation/ and then

        $ sudo apt install automake
        $ sage --pip install g6k

    You should be able to

        $ sage: import g6k

    without error
