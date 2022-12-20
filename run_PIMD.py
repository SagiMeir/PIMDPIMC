#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 16:28:44 2022

@author: sagimeir
"""

import numpy as np
import pandas as pd
# import os
# os.chdir('C:/Users/LG/Documents/GitHub/HO-PIMD')
from sim import Simulation
# in hartree unit system
hbar=1.
BOLTZMANN=3.166811563E-6 
K_harm=1.21647924E-8
mass=1
omega=np.sqrt(K_harm/mass)

parms={"omega": omega}
T = hbar * omega / BOLTZMANN / 6 # temperature

#one atomic unit time in hartree units is 2.4188843265857(47)×10−17 sec

mysim = Simulation( dt=100/2.4188843265857, L=11.3E+2 , temp=T, Nsteps=10000, 
                        mass=1, R=np.zeros( (1,3) ),beads=4, ftype="Harm", 
                        kind=["Ar"], printfreq=100, fac=1)

# mysim = Simulation( dt=100/2.4188843265857, L=11.3E+2 , temp=T, Nsteps=100000, 
#                        mass=1, R=np.zeros( (1,3) ),beads=128, ftype="Harm", 
#                        kind=["Ar"], printfreq=100, fac=1,thermo_type="Langevin",outname="128sim.log")

# T = hbar * omega / BOLTZMANN / 6
# mysim = Simulation( dt=100/2.4188843265857, L=11.3E+2 , temp=T, Nsteps=100000, 
#                         mass=1, R=np.zeros( (1,3) ),beads=32, ftype="Harm", 
#                         kind=["Ar"], printfreq=100, fac=1,thermo_type="Langevin",outname="6hwb_sim.log")


mysim.run(**parms)