# for python3.10 ???
#from collections.abc import Mapping
#from collections.abc import MutableMapping
#from collections.abc import Sequence

# python2 only needs
import sys
from pyQMC.lattice import PolynomialLattice
import numpy as np

# number of dimensions
s = 256 

m = 13
N = 2**m
Pts = np.zeros((N,s))
# name of file containing generating vector
qmc_dir = "/home/z9701564/hdrive/pyQMC_test/pyQMC-master"
filename = qmc_dir+"/staircase2d_spod_a2_C0.1_SPOD_2dstaircase_t0/SPOD_2dstaircase_t0_m"+str(m)+".json"
lat = PolynomialLattice(filename, s)
for k in range(0,N):
    yp = lat.__getitem__(k)
    Pts[k,:] = yp
    print(yp)

# saving as npz file 
outfile = 'SPOD_N'+str(N)+'_dim'+str(s)
np.savez(outfile,P=Pts)
