import sys,os

sys.path.append('../../StarSampler/StarSampler')

import star_sampler as ssp
import osipkov_merritt as om
import scipy as sp

from contextlib import contextmanager
import numpy as np
import warnings

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

stat,error=float(sys.argv[1]),float(sys.argv[2])

model_param = {'ra': 0.1, 'rs_s':0.1, 'al_s':2, 'be_s':5, 'ga_s':.1,
               'rho':.064*1e9, 'rs':1.0, 'alpha':1., 'beta':3., 'gamma':1.}
Nstars= 14700

#1. construct the OM model
#model_param = {'ra': 0.1, 'rs_s':0.1, 'al_s':2, 'be_s':5., 'ga_s':0,
#               'rho':.064*1e9, 'rs':0.1, 'alpha':2., 'beta':5., 'gamma':0}


model_param = {'ra': 1000000, 'rs_s':0.196, 'al_s':2, 'be_s':5., 'ga_s':0.,
               'rho':230070486, 'rs':0.351, 'alpha':1., 'beta':3., 'gamma':1.}

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    with suppress_stdout():
        om1 = om.OM(**model_param)
        
        #2. using rejection sampling
        x1,y1,z1,vx1,vy1,vz1 = ssp.rejection_sample(om1, samplesize = Nstars,
                        r_vr_vt=True, filename=None)
                        
        #3, Or use importance sampling.
        #x2,y2,z2,vx2,vy2,vz2 = ssp.impt_sample(om1, steps=20, resample_factor=5,
        #                samplesize = Nstars, replace=True, r_vr_vt=True, filename=None)

'''
q=(zip(x1,y1,z1,vx1,vy1,vz1))
for i in q:
    print(*i)
'''
r=np.sqrt(x1**2+y1**2)

vsys=np.random.normal(0,stat,len(vz1))
if stat<=0:
    vsys=0

vlos=vz1+vsys
#vmeaserr=np.random.normal(0, 3, len(vz1))+vsys
def gaussian(x,a,b,c):
    return a*np.exp(-(x-b)**2/(2*c**2))
#xes=np.linspace(0,4.5,10000)
#pdf=gaussian(xes,0.42,0.1,0.2)+gaussian(xes,0.28,3.2,1.5)
#cdf=np.cumsum(pdf)
#cdf/=cdf[-1]
#cdf[0]=0
#icdf=sp.interpolate.interp1d(cdf,xes)
#vmeaserr=icdf(np.random.uniform(0,1,len(vz1)))
vmeaserr=np.full_like(vz1,error)

q=zip(r,vlos,vmeaserr)
print('#RPROJ   VMEAS   VERR')
for i in q:
    print(*i)
