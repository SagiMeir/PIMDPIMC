#imports
import numpy as np
import pandas as pd
# from scipy.constants import Boltzmann as BOLTZMANN
# from scipy.constants import hbar 
# for hartree unit system the constansts are:
hbar=1.
BOLTZMANN=3.166811563E-6 
#import matplotlib.pyplot as plt

class Simulation:
    
    def __init__( self, dt, L, temp=298, M=4, Nsteps=0, R=None, beads=1, mass=None, kind=None, \
                 p=None, F=None, U=None, K=None, seed=937142, ftype=None, \
                 step=0, printfreq=1000, xyzname="sim.xyz", fac=1.0, thermo_type=None, \
                 outname="sim.log", debug=False ):
        """
        THIS IS THE CONSTRUCTOR. SEE DETAILED DESCRIPTION OF DATA MEMBERS
        BELOW. THE DESCRIPTION OF EACH METHOD IS GIVEN IN ITS DOCSTRING.
        Parameters
        ----------
        dt : float
            Simulation time step.
            
        L : float
            Simulation box side length.
      
        temp: float
            The temperature.
        
        M: int
            Number of Nose-Hoover thermostates
            
        Nsteps : int, optional
            Number of steps to take. The default is 0.
            
        R : numpy.ndarray, optional
            Particles' positions, Natoms x 3 array. The default is None.
            
        beads : int, optional
            number of ring-polymer beads for the PIMD/PIMC simulation. default is 1.
            
        mass : numpy.ndarray, optional
            Particles' masses, Natoms x 1 array. The default is None.
            
        kind : list of str, optional
            Natoms x 1 list with atom type for printing. The default is None.
            
        p : numpy.ndarray, optional
            Particles' momenta, Natoms x 3 array. The default is None.
            
        F : numpy.ndarray, optional
            Particles' forces, Natoms x 3 array. The default is None.
            
        U : float, optional
            Potential energy . The default is None.
            
        K : numpy.ndarray, optional
            Kinetic energy. The default is None.
            
        seed : int, optional
            Big number for reproducible random numbers. The default is 937142.
            
        ftype : str, optional
            String to call the force evaluation method. The default is None.
            
        step : INT, optional
            Current simulation step. The default is 0.
            
        printfreq : int, optional
            PRINT EVERY printfreq TIME STEPS. The default is 1000.
            
        xyzname : TYPE, optional
            DESCRIPTION. The default is "sim.xyz".
            
        fac : float, optional
            Factor to multiply the positions for printing. The default is 1.0.
        
        thermo_type: str, optional
            String to call the thermostating evaluation method. The default is None.
            
        outname : TYPE, optional
            DESCRIPTION. The default is "sim.log".
            
        debug : bool, optional
            Controls printing for debugging. The default is False.
        Returns
        -------
        None.
        """
        
        #general        
        self.debug=debug 
        self.printfreq = printfreq 
        self.xyzfile = open( xyzname, 'w' ) 
        self.outfile = open( outname, 'w' ) 
        
        #simulation
        self.temp=temp
        self.Nsteps = Nsteps 
        self.dt = dt 
        self.L = L 
        self.seed = seed 
        self.step = step         
        self.fac = fac
        self.M=M # number of NH thermostates
        self.beads=beads
        self.spring_E=0 # the springs energy
        self.spring_F=0 # springs force
        # self.Q1=Q1
        
        #system        
        if R is not None:
            self.R = R        
            self.mass = mass
            self.kind = kind
            self.Natoms = self.R.shape[0]
            self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
            self.Rbeads = np.zeros( (self.Natoms,3,self.beads) )
        else:
            self.R = np.zeros( (1,3) )
            self.mass = 1.6735575E-27 #H mass in kg as default
            self.kind = ["H"]
            self.Natoms = self.R.shape[0]
            self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
            self.Rbeads = np.zeros( (1,3,self.beads) )
        if p is not None:
            self.p = p
            self.K = K
        else:
            self.p = np.zeros( (self.Natoms,3,self.beads) )
            self.K = 0.0
        
        if F is not None:
            self.F = F
            self.U = U
        else:
            self.F = np.zeros( (self.Natoms,3,self.beads) )
            self.U = 0.0
        
        # the ring polymer freq w_P:
        self.omega_P = np.sqrt(self.beads) * BOLTZMANN * self.temp / hbar
        # the thermostates masses
        self.Q = BOLTZMANN * self.temp / self.omega_P**2
        # stagging trasformation
        self.u=np.zeros((self.Natoms,3,self.beads))
        
        # stagging masses
        self.mk = np.zeros((self.Natoms,3,self.beads))
        self.mk_prime = np.zeros((self.Natoms,3,self.beads))
        self.mk[:,:,0] = 0
        self.mk_prime[:,:,0] = self.mass
        for i in range(1,self.beads):
            self.mk[:,:,i] = (i+1)*self.mass/i
            self.mk_prime[:,:,i] = (i+1)*self.mass/i
        
        
        #set RNG seed
        np.random.seed( self.seed )
        
        #check force type
        if ( ftype == "LJ" or ftype == "Harm" or ftype == "Anharm"):
            self.ftype = "eval" + ftype
            if (ftype == "LJ"):
                self.Utype = "eval_delta" + ftype
            else:
                self.Utype = "eval" + ftype
                self.deltaLJ=0
        else:
            raise ValueError("Wrong ftype value - use LJ or Harm or Anharm.")
        
        #check the thermostating type
        if (thermo_type == "NHC" or thermo_type == "Langevin"):
            self.thermo_type="Thermostat_"+thermo_type
        elif thermo_type == None:
            self.thermo_type="Nothing"
        else:
            raise ValueError("Wrong thermo_type value - use NHC or Langevin.")
    
    def __del__( self ):
        """
        THIS IS THE DESCTRUCTOR. NOT USUALLY NEEDED IN PYTHON. 
        JUST HERE TO CLOSE THE FILES.
        Returns
        -------
        None.
        """
        self.xyzfile.close()
        self.outfile.close()
    
    def evalForce( self, **kwargs ):
        """
        THIS FUNCTION CALLS THE FORCE EVALUATION METHOD, BASED ON THE VALUE
        OF FTYPE, AND PASSES ALL OF THE ARGUMENTS (WHATEVER THEY ARE).
        Returns
        -------
        None. Calls the correct method based on self.ftype.
        """
        
        getattr(self, self.ftype)(**kwargs)
    
    def evalPotential( self, **kwargs ):
        """
        THIS FUNCTION CALLS THE Potential EVALUATION METHOD for the MC simulation, BASED ON THE VALUE
        OF UTYPE, AND PASSES ALL OF THE ARGUMENTS (WHATEVER THEY ARE).
        Returns
        -------
        None. Calls the correct method based on self.ftype.
        """
        
        getattr(self, self.Utype)(**kwargs)
    
    def Thermostat(self):
        """
        THIS FUNCTION CALLS THE DESIRED THERMOSTATING METHOD.

        Returns
        -------
        None. Calls the correct method based on self.thermo_type

        """
        getattr(self, self.thermo_type)
        
    def dumpThermo( self ):
        """
        THIS FUNCTION DUMPS THE ENERGY OF THE SYSTEM TO FILE.
        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py
        Returns
        -------
        None.
        """
        if( self.step == 0 ):
            self.outfile.write( "step K U E \n" )
        
        self.outfile.write( str(self.step) + " " \
                          + "{:.6e}".format(self.K) + " " \
                          + "{:.6e}".format(self.U) + " " \
                          + "{:.6e}".format(self.E) + "\n" )
                
    def dumpXYZ( self ):
        """
        THIS FUNCTION DUMP THE COORDINATES OF THE SYSTEM IN XYZ FORMAT TO FILE.
        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py
        Returns
        -------
        None.
        """
            
        self.xyzfile.write( str( self.Natoms ) + "\n")
        self.xyzfile.write( "Step " + str( self.step ) + "\n" )
        
        for i in range( self.Natoms ):
            self.xyzfile.write( self.kind[i] + " " + \
                              "{:.6e}".format( self.R[i,0]*self.fac ) + " " + \
                              "{:.6e}".format( self.R[i,1]*self.fac ) + " " + \
                              "{:.6e}".format( self.R[i,2]*self.fac ) + "\n" )
    
    def readXYZ( self, inpname ):
        """
        THIS FUNCTION READS THE INITIAL COORDINATES IN XYZ FORMAT.
        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py
        Returns
        -------
        None.
        """
           
        df = pd.read_csv( inpname, sep="\s+", skiprows=2, header=None )
        
        self.kind = df[ 0 ]
        self.R = df[ [1,2,3] ].to_numpy()
        self.Natoms = self.R.shape[0]
        
    def Nothing(self):
        """
        FUNCTION THAT DOES NOTHING. CORRENTLY USFULL FOR THE NVE ENSEMBLE
        """
        
        
        
################################################################
################## NO EDITING ABOVE THIS LINE ##################
################################################################
    
    
    def sampleMB( self, removeCM=True ):
        """
        THIS FUNCTIONS SAMPLES INITIAL MOMENTA FROM THE MB DISTRIBUTION.
        IT ALSO REMOVES THE COM MOMENTA, IF REQUESTED.
        Parameters
        ----------
        removeCM : bool, optional
            Remove COM velocity or not. The default is True.
        Returns
        -------
        None. Sets the value of self.p.
        """
        
        # for every dimansion P(v)=sqrt(m/2piKT)*exp(-mv^2/2KT)
        # This is a normal distribution with mean=0, sigma=sqrt(KT/m)
        self.p= self.mass * np.random.normal(loc=0,
                                             scale=(BOLTZMANN*self.temp/self.mass)**0.5,
                                             size=(self.Natoms,3,self.beads)
                                             )    
        # loc is the mean, scale is sigma, and size is N dimansional
        if removeCM:
            self.p[:,0,:]-=self.p.mean(axis=(0,2))[0]
            self.p[:,1,:]-=self.p.mean(axis=(0,2))[1]
            self.p[:,2,:]-=self.p.mean(axis=(0,2))[2]
                
    def sampleMB_NoseHoover( self ):
        """
        THIS FUNCTIONS SAMPLES INITIAL MOMENTA FROM THE MB DISTRIBUTION.
        IT ALSO REMOVES THE COM MOMENTA, IF REQUESTED.
        Parameters
        ----------
        temp : float
            The temperature to sample from.
        removeCM : bool, optional
            Remove COM velocity or not. The default is True.
        Returns
        -------
        None. Sets the value of self.p.
        """
        
        # for every dimansion P(v)=sqrt(m/2piKT)*exp(-mv^2/2KT)
        # This is a normal distribution with mean=0, sigma=sqrt(KT/m)
        self.p_eta= self.Q * np.random.normal(loc=0,
                                             scale=(BOLTZMANN*self.temp/self.Q)**0.5,
                                             size=(self.Natoms,self.M)
                                             )     
        # loc is the mean, scale is sigma, and size is M*N dimansional
        
        # the "forces" of the nose-hoover chain
        self.F_eta=np.zeros([self.Natoms,self.M])
        # the "positions"
        self.eta=np.zeros([self.Natoms,self.M])
    
    def applyPBC( self ):
        """
        THIS FUNCTION APPLIES PERIODIC BOUNDARY CONDITIONS.
        Returns
        -------
        None. Sets the value of self.R.
        """
        
        self.R[self.R>(self.L /2)]-=self.L
        self.R[self.R<=-(self.L /2)]+=self.L
                    
    def removeRCM( self ):
        """
        THIS FUNCTION ZEROES THE CENTERS OF MASS POSITION VECTOR.
        Returns
        -------
        None. Sets the value of self.R.
        """    

        x_cm=self.R[:,0].mean()
        y_cm=self.R[:,1].mean()
        z_cm=self.R[:,2].mean()
        self.R[:,0]=self.R[:,0]-x_cm
        self.R[:,1]=self.R[:,1]-y_cm
        self.R[:,2]=self.R[:,2]-z_cm
    
             
    def evalLJ( self, eps, sig ):
        """
        THIS FUNCTION EVALUTES THE LENNARD-JONES POTENTIAL AND FORCE.
        Parameters
        ----------
        eps : float
            epsilon LJ parameter.
        sig : float
            sigma LJ parameter.
        Returns
        -------
        None. Sets the value of self.F and self.U.
        """
        
        if( self.debug ):
            print( "Called evalLJ with eps = " \
                  + str(eps) + ", sig= " + str(sig)  )
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################

     
        
        def r_ij(ri,rj):
            '''
            returns ri-rj vector with PBC.
            The i-th particle is influenced by j-th particle in range up to L/2
            Using mode with L/2 ensures that the i-th particle is centered inside L*L*L box
            '''
            rij=ri-rj
            rij[rij>(self.L /2)] -=self.L
            rij[rij<=-(self.L /2)] +=self.L
            return rij
                
        def r_ij_mat(r_particle_vec):
            rij=r_particle_vec[:, np.newaxis, :] - r_particle_vec[np.newaxis, :, :]
            rij[rij>(self.L /2)] -=self.L
            rij[rij<=-(self.L /2)] +=self.L
            return rij
        def term_in_the_sum(r_squared):
            '''
            Getting the squared distance between two particles and returns
            one term in the LJ sum for the potential
            '''
            sig6overR6 = sig**6 * r_squared**(-3) # one term in the equation
            return 4*eps*sig6overR6*(sig6overR6-1)
        def term_in_Force_sum(r_squared):
            '''
            Getting the squared distance between two particles and returns
            one term in the LJ sum for the Force.
            Note that the return need to be multiplyed by the r_ij vector
            '''
            sig6overR6 = sig**6 * r_squared**(-3)
            sig8overR8 = sig**8 * r_squared**(-4)
            
            return ( 24*eps*(sig**(-2)) ) * sig8overR8 * (2*sig6overR6-1)

            
        ri_minus_rj=r_ij_mat(self.R)
        rij_squared=(ri_minus_rj**2).sum(-1) # This way I sum only on the x,y,z components
        rij_squared_triu=np.triu(rij_squared.copy()) # making a triungular matric
        U_LJ=term_in_the_sum(rij_squared_triu[rij_squared_triu>0.]).sum()
        # Below I modify the diagonal of rij_squared to avoid deviding in zero 
        rij_squared_modif= rij_squared + np.diag(np.ones(self.Natoms)*10000)
        Fx = (term_in_Force_sum(rij_squared_modif)*ri_minus_rj[:,:,0]).sum(-1)
        Fy = (term_in_Force_sum(rij_squared_modif)*ri_minus_rj[:,:,1]).sum(-1)
        Fz = (term_in_Force_sum(rij_squared_modif)*ri_minus_rj[:,:,2]).sum(-1)
        
        F_LJ=(np.vstack((Fx,Fy,Fz))).transpose()
        
        self.U=U_LJ
        self.F=F_LJ.copy()
        
            
    def eval_deltaLJ( self, eps, sig ):
        """
        THIS FUNCTION EVALUTES THE change in the LENNARD-JONES POTENTIAL 
        as a risult of moving the k-th particle.
        Parameters
        ----------
        eps : float
            epsilon LJ parameter.
        sig : float
            sigma LJ parameter.
        Returns
        -------
        None. Sets the value and self.deltaLJ.
        """
        
        
        r_ij=self.R[self.kth,:]-self.R
        r_ij[r_ij>(self.L /2)] -=self.L
        r_ij[r_ij<=-(self.L /2)] +=self.L
        r_squared=(r_ij**2).sum(-1)
        r_squared=r_squared[r_squared>0]
        sig6overR6 = sig**6 * r_squared**(-3)
        self.deltaLJ = (4*eps*sig6overR6*(sig6overR6-1)).sum()
         
        
    
    def evalHarm( self, omega ):
        """
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR A HARMONIC TRAP.
        Parameters
        ----------
        omega : float
            The frequency of the trap.
        Returns
        -------
        None. Sets the value of self.F, self.U and self.K
        """
        
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
        P=self.beads
        # self.F[:,:,0] = - self.mass * omega**2 * ( self.Rbeads.sum(axis=(2)) ) / P
        # for i in range(1,P):
            # self.F[:,:,i] = - self.mass * omega**2 * self.Rbeads[:,:,i]/P   + ( (i-1)/i) *  self.F[:,:,i-1] 
        self.F= - self.mass * omega**2 *  self.Rbeads  / P 
        self.U= 0.5 * self.mass * omega**2 * ( (self.Rbeads)**2 ).sum() / P 
        # x_c=self.Rbeads[:,0,:].mean()
        # self.K = 0.5*( BOLTZMANN * self.temp + ( (self.Rbeads[:,0,:]-x_c)*self.mass*omega**2 * self.Rbeads[:,0,:] ).sum()/P )


    def evalNoseHoover( self ):
        """
        evaluated the nose-hoover 'forces'
        Returns
        -------
        None.
        """
        
        F_eta0= self.p**2/self.mk_prime - BOLTZMANN*self.temp
        F_eta_i = self.p_eta[:,:-1]**2/self.Q - BOLTZMANN*self.temp
        self.F_eta=np.vstack((F_eta0.transpose(),F_eta_i.transpose())).transpose()
        
        

    def StaggingTrans( self ):
        """
        Transforms the position coordinates to the stagging coordinates.
        Returns
        -------
        None. sets the values of self.u
        """
        P=self.beads
        self.u[:,:,0]=self.Rbeads[:,:,0]
        for i in range(1,P):
            self.u[:,:,i]=self.Rbeads[:,:,i] - ( i*self.Rbeads[:,:,(i+1)%P] + self.Rbeads[:,:,0] ) / (i+1)
        # self.u[:,:,1:] = self.Rbeads[:,:,1:] - (np.arange(1,P)[np.newaxis,np.newaxis,:]*self.Rbeads[:,:,(np.arange(1,P)+1)%P] + self.Rbeads[:,:,0] )/(np.arange(1,P)+1)[np.newaxis,np.newaxis,:]
    
    
    def Inverse_Stagging( self ):
        """
        Transforms back stagging coordinates to the the position coordinates.
        Returns
        -------
        None. sets the values of self.R
        """
        P=self.beads
        self.Rbeads[:,:,0]=self.u[:,:,0]
        for i in range(P,1,-1):
            self.Rbeads[:,:,i-1] = self.u[:,:,i-1] + (i-1)*self.Rbeads[:,:,i%P]/i +  self.u[:,:,0]/i
        # self.Rbeads[:,:,1:] = self.u[:,:,0] + ( np.arange()*self.u[:,:,1:] ).sum()
    
    
    def CalcKinE( self ):
        """
        THIS FUNCTIONS EVALUATES THE KINETIC ENERGY OF THE SYSTEM.
        Returns
        -------
        None. Sets the value of self.K.
        """
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
        self.K =( ((self.p)**2).sum() )/ (2*self.mass)
    
    def spring_energy( self ):
        """
        THIS FUNCTION CALCULATES THE ENERGY AND FORCES OF THE SPRINGS BETWEEN THE BEADS

        Returns
        -------
        None. Sets self.spring_E, self.spring_F

        """
        P=self.beads
        x_next=self.Rbeads[:,:,(np.arange(P)+1)%P]
        x_prev=self.Rbeads[:,:,(np.arange(P)-1)%P]
        self.spring_F=-self.mass*self.omega_P**2*(2*self.Rbeads-x_next-x_prev)
        self.spring_E=0.5*self.mass*self.omega_P**2*((x_next-self.Rbeads)**2).sum()
        # self.spring_E=0
        # self.spring_F=np.zeros((self.Natoms,3,P))
        # for j in range(self.Natoms):
            # for i in range(P):
                # self.spring_E+=0.5*self.mass*self.omega_P**2*((self.Rbeads[j,:,(i+1)%P]-self.Rbeads[j,:,i])**2).sum() 
                # self.spring_F[j,:,i]=-self.mass*self.omega_P**2*(2*self.Rbeads[j,:,i]-self.Rbeads[j,:,(i+1)%P]-self.Rbeads[j,:,(i-1)%P]) 
    
    def Thermostat_Langevin( self ):
        """
        THIS FUNCTION THERMOSTATES THE SYSTEM TO THE DESIRED TEMPRERATURE.

        Returns
        -------
        None.

        """
        # self.p =( self.p * np.exp(-0.5*self.omega_P*self.dt) + 
        #           (self.mk*self.temp*BOLTZMANN*(1-np.exp(-self.omega_P*self.dt)))**0.5 * 
        #           np.random.normal(loc=0,scale=1,size=(self.Natoms))  )
        self.p =( self.p * np.exp(-0.5*self.omega_P*self.dt) + 
                  (self.mass*self.temp*BOLTZMANN*(1-np.exp(-self.omega_P*self.dt)))**0.5 * 
                  np.random.normal(loc=0,scale=1,size=(self.Natoms,3,self.beads))  )
        
        
    def VVstep( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE VELOCITY VERLET STEP.
        Returns
        -------
        None. Sets self.R, self.p.
        """

        # The steps are written in detail in the jupyter notebook report.
        self.Thermostat()
        
        self.spring_energy()
        # self.p+= 0.5*(-self.mk*self.omega_P**2*self.u + self.F)*self.dt
        self.p+= 0.5*(self.spring_F + self.F )*self.dt 
        
        self.Rbeads+=self.p*self.dt/self.mass
        # self.u+= self.p*self.dt/self.mk_prime
        # self.Inverse_Stagging()
        self.R=self.Rbeads.mean(axis=(2))
        self.evalForce(**kwargs)
        self.spring_energy()
        
        # self.p+= 0.5*(-self.mk*self.omega_P**2*self.u + self.F)*self.dt
        self.p+= 0.5*(self.spring_F + self.F )*self.dt 
        
        self.Thermostat()
       
        ''' 
        # STEP 1:
        p_eta_M = self.p_eta[:,self.M-1] + 0.25*self.dt*self.F_eta[:,self.M-1]
        p_eta_i= ( self.p_eta[:,:self.M-1]*np.exp(-0.25*self.dt*self.p_eta[:,1:]/self.Q) + 
                  self.F_eta[:,:self.M-1]*self.Q*(1-np.exp(-0.25*self.dt*self.p_eta[:,1:]/self.Q))/self.p_eta[:,1:] )
        self.p_eta=np.vstack((p_eta_i.transpose(),p_eta_M.transpose())).transpose()
        # STEP 2:
        self.p*= np.exp(-0.5*self.dt*self.p_eta[:,0]/self.Q)
        self.eta+=0.5*self.dt*self.p_eta/self.Q
        self.evalNoseHoover()
        # STEP 3:
        p_eta_M = self.p_eta[:,self.M-1] + 0.25*self.dt*self.F_eta[:,self.M-1]
        p_eta_i= ( self.p_eta[:,:self.M-1]*np.exp(-0.25*self.dt*self.p_eta[:,1:]/self.Q) + 
                  self.F_eta[:,:self.M-1]*self.Q*(1-np.exp(-0.25*self.dt*self.p_eta[:,1:]/self.Q))/self.p_eta[:,1:] )
        
        # STEP 4:
        self.p+= 0.5*(-self.mk*self.omega_P**2*self.u + self.F)*self.dt
        # STEP 5:
        self.u+= self.p*self.dt/self.mk_prime
        self.Inverse_Stagging()
        self.evalForce(**kwargs)
        # STEP 6:
        self.p+= 0.5*(-self.mk*self.omega_P**2*self.u + self.F)*self.dt
        
        self.evalNoseHoover()
        # STEP 7:
        p_eta_M = self.p_eta[:,self.M-1] + 0.25*self.dt*self.F_eta[:,self.M-1]
        p_eta_i= ( self.p_eta[:,:self.M-1]*np.exp(-0.25*self.dt*self.p_eta[:,1:]/self.Q) + 
                  self.F_eta[:,:self.M-1]*self.Q*(1-np.exp(-0.25*self.dt*self.p_eta[:,1:]/self.Q))/self.p_eta[:,1:] )
        self.p_eta=np.vstack((p_eta_i.transpose(),p_eta_M.transpose())).transpose()
        # STEP 8:
        self.p*= np.exp(-0.5*self.dt*self.p_eta[:,0]/self.Q)
        self.eta+=0.5*self.dt*self.p_eta/self.Q
        self.evalNoseHoover()
        # STEP 9:
        p_eta_M = self.p_eta[:,self.M-1] + 0.25*self.dt*self.F_eta[:,self.M-1]
        p_eta_i= ( self.p_eta[:,:self.M-1]*np.exp(-0.25*self.dt*self.p_eta[:,1:]/self.Q) + 
                  self.F_eta[:,:self.M-1]*self.Q*(1-np.exp(-0.25*self.dt*self.p_eta[:,1:]/self.Q))/self.p_eta[:,1:] )
        
        '''
        
    def run( self, **kwargs ):
        """
        THIS FUNCTION DEFINES A SIMULATION DOES, GIVEN AN INSTANCE OF 
        THE SIMULATION CLASS. YOU WILL NEED TO:
            1. EVALUATE THE FORCES (USE evaluateForce() AND PASS A DICTIONARY
                                    WITH ALL THE PARAMETERS).
            2. PROPAGATE FOR NS TIME STEPS USING THE VELOCITY VERLET ALGORITHM.
            3. APPLY PBC.
            4. CALCULATE THE KINETIC, POTENTIAL AND TOTAL ENERGY AT EACH TIME
            STEP. 
            5. YOU WILL ALSO NEED TO PRINT THE COORDINATES AND ENERGIES EVERY 
        PRINTFREQ TIME STEPS TO THEIR RESPECTIVE FILES, xyzfile AND outfile.
        Returns
        -------
        None.
        """      
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################ 
        #E_list=[]
        #U_list=[]
        #K_list=[]
        H_list=[] # the classical energy
        spring_E_list=[]
        step_list=[]
        # self.StaggingTrans()
        self.sampleMB(removeCM=False)
        # self.sampleMB_NoseHoover()
        for self.step in range(self.Nsteps):
            # self.Inverse_Stagging()
            self.evalForce(**kwargs)
            # self.evalNoseHoover()
            self.VVstep(**kwargs)
            # self.H=( (0.5*self.p**2/self.mk_prime +0.5*self.omega_P**2*self.mk*self.u**2).sum()
            #         + self.Natoms*self.U +  0.5*(self.p_eta**2).sum()/self.Q 
            #         + BOLTZMANN*self.temp*self.eta.sum()) 
            # self.H=  (0.5*self.p**2/self.mk_prime +0.5*self.omega_P**2*self.mk*self.u**2).sum() + self.U
            self.H=  (0.5*self.p**2/self.mass).sum() +self.spring_E + self.U
            #self.applyPBC()
            self.CalcKinE()
            # the potential and kinetic energy was estimated at VVstep that called evalForce
            self.E=self.K+self.U
            if (( (self.step)%(self.printfreq) )==0):
                #E_list.append(self.E)
                #U_list.append(self.U)
                #K_list.append(self.K)
                step_list.append(self.step)
                H_list.append(self.H)
                spring_E_list.append(self.spring_E)
                self.dumpThermo()
                self.outfile.flush()
                self.dumpXYZ()
                self.xyzfile.flush()
        simlog=pd.DataFrame({'step':np.array(step_list),'H':np.array(H_list),'sE':np.array(spring_E_list)})
        #simlog=pd.DataFrame({'step':np.array(step_list),'K':np.array(K_list),'U':np.array(U_list),'E':np.array(E_list)})
        simlog.to_csv('simlog.csv',index=False)

    
    def MCstep( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE METROPOLIS MC STEP IN THE NVT ENSEMBLE.
        YOU WILL NEED TO PROPOSE TRANSLATION MOVES, APPLY  
        PBC, CALCULATE THE CHANGE IN POTENTIAL ENERGY, ACCEPT OR REJECT, 
        AND CALCULATE THE ACCEPTANCE PROBABILITY.
        Returns
        -------
        None. Sets self.R.
        """
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
        
        
        self.kth=np.random.randint(0,self.Natoms) # chosing the k-th particle to move
        self.evalForce(**kwargs)
        
        old_u=(self.u).copy()
        old_V=self.U + 0.5*self.omega_P**2*(self.mk*self.u**2).sum()
        # old_R = (self.R).copy()
        # x_next = np.append(self.R[1:,0],self.R[0,0])
        # old_V = self.U + 0.5*self.omega_P**2*self.mass*((x_next-self.R[:,0])**2).sum()
        
        
        # The suggested move
        self.delta=self.dt*np.random.uniform(-1,1,size=(1)) 
        # self.R[self.kth,0]+=self.delta
        # x_next = np.append(self.R[1:,0],self.R[0,0])
        self.u[self.kth]+=self.delta
        self.Inverse_Stagging()
        self.evalForce(**kwargs)
        
        deltaV = self.U + 0.5*self.omega_P**2*(self.mk*self.u**2).sum() - old_V
        # deltaV = self.U + 0.5*self.omega_P**2*self.mass*((x_next-self.R[:,0])**2).sum() - old_V
        
        
        # MCMC condition:
        if np.exp(-deltaV/(BOLTZMANN*self.temp))>np.random.uniform():
            return 1
        else:
            # self.R = old_R.copy()
            self.u = old_u.copy()
            self.Inverse_Stagging()
            self.evalForce(**kwargs)
            return 0
                


    def runMC( self, **kwargs ):
        """ 
        THIS FUNCTION DEFINES AN MC SIMULATION DOES, GIVEN AN INSTANCE OF 
        THE SIMULATION CLASS. YOU WILL NEED TO LOOP OVER MC STEPS, 
        PRINT THE COORDINATES AND ENERGIES EVERY PRINTFREQ TIME STEPS 
        TO THEIR RESPECTIVE FILES, SIMILARLY TO YOUR MD CODE.
        Returns
        -------
        None.
        """   
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
        self.accept=0
        self.evalForce(**kwargs)
        self.StaggingTrans()
        for self.step in range(self.Nsteps):
            self.accept+=self.MCstep(**kwargs)
            
            
            if (( (self.step)%(self.printfreq) )==0):
                self.E=self.U+self.K
                self.dumpThermo()
                self.outfile.flush()
                
                self.dumpXYZ()
                self.xyzfile.flush()
