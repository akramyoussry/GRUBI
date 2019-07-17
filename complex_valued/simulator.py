"""
This module implements the chip model via the FourElectrodeArraySimulator.
"""

# preample
import numpy as np
from scipy.signal import lti
from scipy.linalg import expm,block_diag
from itertools import accumulate

class FourElectrodeArraySimulator(object):
    """
    Class for simulating the waveguide chip ==> This is a modified version of Robert's original implementation. The modifcations are:
    -- added the functionality of having an input time-varying input voltage
    -- added a model for drifiting by defining an lti object to calculate the actual response of the input voltage before passing to the Hamiltonian
    -- added a model for coupling loss of the detector fibres by modeling a non-unitary transformation on the output probabilties
    -- replacing the np.matrix instances with np.array
    -- added comments and docstrings 
    """

    def __init__(self, num_wg, coupling_coeff, n0=2.1753, delta_n=5e-6,
                 c0=100., delta_c1=1.5, delta_c2=-1.3, wavelength=808.e-9, fan_length=0.0, 
                 circuit_params=[70.0, 70.0, 700.0, 700.0, 30e3, 1e3, 10e-6, 25e-6], length=3.6e-2 ):
        """
        Class constructor
        
        num_wg: number of waveguides of the chip
        coupling_coeff    : an array of length num_wg who entries represent the coupling coefficients modeling optical fibre power losses at he outputs
        n0                : refractive index of the substrate
        delta_n           : dynamic proportionality constant for the propagation constant as a linear function of the volts across the waveguide
        C0                : intrinsic waveguide coupling
        delta_C1, delta_C2: dynamic proportinality constants for the coupling between waveguides as a linear function of the volts accross the waveguides
        wavelength        : operating wavelength of the chip
        fan_length        : the fan length of the waveguides on the chip
        circuit_params    : the values of R1,R2,R3,R4,R5,R6,C1,C2 of the equivalent circuit model of the chip
        length            : the length of the waveguide along which the photon will propagate 
        """
        
        # store the different parameters
        self.num_wg = num_wg
        self.num_electrodes = 2 * num_wg
        self.n0 = n0
        self.delta_n = delta_n
        self.beta0 = [2 * np.pi * n0 / wavelength for _ in range(num_wg)]
        self.c0 = [(1j)*c0 for _ in range(num_wg - 1)]
        self.delta_c1 = delta_c1
        self.delta_c2 = delta_c2
        self.wavelength = wavelength
        self.fan_length = fan_length
        self.circuit_params = circuit_params
        self.coupling_coeff = coupling_coeff
        self.length  = length
        self.betas = self.beta0
        self.cs = self.c0
                
        # initialize zero-voltage Hamiltonian
        self.set_hamiltonian()
        
        # initialize fan-out Hamiltonian
        self.set_fan_hamiltonian()
        
        # initialize the input waveguide to the first one. This can be modified externally when using the class
        self.set_input_waveguide(0)

    def set_voltages(self, voltages):
        """
        This method to calculate the matrix elements of the total Hamiltonian.
        
        voltages: an array of size equal to the number of electrodes storing the values of voltages applied to each electrode
        """
        
        # calculate the diagonal elements of the Hamiltonian representing propagation constant of each waveguide
        bss = [(2 * np.pi / self.wavelength) *
                      (self.delta_n * (voltages[i + 1] - voltages[i])) for i in range(0, self.num_electrodes, 2)]

        self.betas = [b + s for b, s in zip(self.beta0, bss)]

        # calculate the off-diagonal elements of the total Hamiltonian representing the coupling between waveguides 
        css = [self.delta_c1 * (voltages[i + 2] - voltages[i + 1]) +
               self.delta_c2 * ((voltages[i + 1] - voltages[i]) + (voltages[i + 3] - voltages[i + 2])) for i in
               range(0, self.num_electrodes - 2, 2)]

        self.cs = [c + s for c, s in zip(self.c0, css)]
    

    def ct_to_tf(self, R1,R2,R3,R4,R5,R6,C1,C2):
        """
        This is a private method to convert circuit parameters to transfer function coefficients in order to simulate the equivalent circuit model of the chip
        
        R1, R2, R3, R4, R5, R6 : The values of the resistances
        C1, C2                 : The values of the capacitances
        
        returns a tuple of the two lists encoding the numerator and denominator coefficeints of the transfer function
        """
        
        # calculate the numerator coefficents
        b0 = R3*R5 + R3*R6 + R4*R5 + R4*R6 + R5*R6 
        b1 = C2*R3*R5*R6 + C2*R4*R5*R6
        
        # calculate the denominator coefficients
        a0 = R1*R5 + R1*R6 + R2*R5 + R2*R6 + R3*R5 + R3*R6 + R4*R5 + R4*R6 + R5*R6
        a1 = C1*R1*R3*R5 + C1*R1*R3*R6 + C1*R1*R4*R5 + C1*R1*R4*R6 + C1*R1*R5*R6 + C1*R2*R3*R5 + C1*R2*R3*R6 + C1*R2*R4*R5 + C1*R2*R4*R6 + C1*R2*R5*R6 + C2*R1*R5*R6 + C2*R2*R5*R6 + C2*R3*R5*R6 + C2*R4*R5*R6
        a2 = C1*C2*R1*R3*R5*R6 + C1*C2*R1*R4*R5*R6 + C1*C2*R2*R3*R5*R6 + C1*C2*R2*R4*R5*R6
        
        # return the coefficients grouped into two lists
        num = [b1,b0]
        den = [a2,a1,a0]
        
        return num,den


    def set_hamiltonian(self):
        """
        This method is to construct the Hamiltonian matrix after we calculated the matrix elements by called the method set_voltages
        """
        # initialize the matrix
        h = np.zeros(shape=(self.num_wg, self.num_wg), dtype=np.complex)
        
        # fill the elements in the triadiagonal form
        for i in range(0, self.num_wg - 1):
            h[i, i + 1] = self.cs[i]
            h[i + 1, i] = np.conj(self.cs[i])
            h[i, i] = self.betas[i]
        h[self.num_wg - 1, self.num_wg - 1] = self.betas[self.num_wg- 1]
        
        # store the Hamiltonian
        self.hamiltonian = h
    

    def set_fan_hamiltonian(self):
        """
        This method is to construct the fan Hamiltonian, in this model it is just the zero-voltage Hamiltonian
        """
        
        # initialize the matrix
        h = np.zeros(shape=(self.num_wg, self.num_wg), dtype=np.complex)
        
        # fill the elements in the triadiagonal form
        for i in range(0, self.num_wg - 1):
            h[i, i + 1] = self.c0[i]
            h[i + 1, i] = np.conj(self.c0[i])
            h[i, i] = self.beta0[i]
        h[self.num_wg - 1, self.num_wg - 1] = self.beta0[-1]
        
         # store the fan Hamiltonian
        self.fan_hamiltonian = h
    

    def set_input_waveguide(self, input_waveguide):
        """
        This method to specify the input waveguide. This is modeled through the inital quantum state of the system.
        
        input_waveguide: integer representing the index of the waveguide acting as the input
        """
        
        # initalize a column vector with all zeros
        psi_0 = np.zeros((self.num_wg,1))
        
        # set one element to 1 according to the selected index
        psi_0[input_waveguide] = 1.0
        
        # store the inital state
        self.psi_in = psi_0
        
  
    def evolve(self, length, ideal=0):
        """
        This method to calculate the output probability amplitudes, by evolving the Hamiltonian.
        
        length: the evolution length
        ideal : a parameter that determines whether we model losses at the output or not, with default value 0. 0 results in returning the lossy outputs, while 1 results in returning the amplitudes without losses
        
        returns a column representing the probability amplitudes of the evolved state (with lossed if this was chosen). 
        """
        
        # store the length of evolution
        self.length = length
        
        # construct the Hamiltonian matrix
        self.set_hamiltonian()
        self.set_fan_hamiltonian()

        # construct the evolution matrix
        U_fan = expm(-1j * self.fan_hamiltonian * self.fan_length)
        U = expm(-1j * self.hamiltonian * length)

        # calculate the output state
        psi_1 = U_fan @ self.psi_in
        psi_2 = U @ psi_1
        psi_3 = U_fan @ psi_2
        self.psi_out = psi_3
        
        # calculate the power distribution
        power_distribution = np.abs(psi_3)**2
        
        # calculate the interferometer power distribtuion encoding the phase angles
        power_distribution = np.abs(0.5*(psi_3+1))**2
        interferometer_distribution = np.abs(0.5*(psi_3+1j))**2
        
        # store the outputs
        self.output_distribution = np.concatenate( (power_distribution, interferometer_distribution) ).squeeze()
        
        # calculate the probability amplitudes and relative phase shifts of the output state
        #phi_0 = np.angle(self.psi_out[0])
        #self.output_distribution = np.array( [float(abs(p)) for p in self.psi_out] + [float( (np.angle(p) - phi_0)/(4*np.pi)) for p in self.psi_out[1:]] )
        #self.output_distribution = np.array( [float(np.real(p)) for p in self.psi_out] + [float( np.imag(p) ) for p in self.psi_out] )
        
        # calculate the lossy output power P_loss_i = P_out_i * epsilon_i. The ideal case is epsilon_i = 1 for all i 
        #wg_power = self.output_distribution * np.array(self.coupling_coeff)
        
        # normalize the lossy output power distribution
        #wg_power = wg_power/sum(wg_power)
        
        # check if we need to model the ideal behaviour without losses
        #if ideal==0:
            
            # return the lossy output power distribution
            #return wg_power
        
        #else:
            
            # return the exact output power distribution
        return self.output_distribution
    
  
    def evolve_varying_voltage(self, time_range, input_voltage, length, ideal=0):
        """
        This method is to caluclate the output power distribution given a time-varyig input voltage. This is main method of the simulator.
        
        time_range   : an array of length num_points containing the time points where are we sampling the inputs and outputs
        input_voltage: an array of size (num_points,num_electrodes) storing the voltage at each time instant for each electrode. Each column represents the voltages applied over one electrode, with rows encoding time instants. 
        ideal        : a parameter that determines if we take the equiavalent circuit model into account for the simulation or not. The default value is 0. Use 0 for modeling the circuit effects, 1 for neglecting the circuit effects.     
        
        returns the output waveuide power distribution, in the form of an array of size (num_points,num_wg).
        """
        
        if ideal==0: #use circuit response
            
           # calculate the parameters of the transfer function of the LTI system modeling the chip behaviour
           num,den = self.ct_to_tf(self.circuit_params[0],self.circuit_params[1],self.circuit_params[2],self.circuit_params[3],self.circuit_params[4],self.circuit_params[5],self.circuit_params[6],self.circuit_params[7])
           
           # define an lti obejct tp simulate the circuit
           circuit = lti(num,den)
           
           # evaluate the circuit response of the input voltage 
           voltages_w_decay = np.array([ circuit.output(v,time_range)[1] for v in input_voltage.T]).T
        
        else: # use ideal pulses
           
            # do not transform the input
           voltages_w_decay = input_voltage
        
        
        # evaluate the output distribuion by doing the quantum evolution at each time instant-->
        
        # initialize the array to store the power distribution 
        wg_power         = np.zeros((time_range.size,2*self.num_wg)) 
        
        # loop over all time instants
        for idt,t in enumerate(time_range):
            
            # calcualte the voltage-dependent parameters of the Hamilontian
            self.set_voltages(voltages_w_decay[idt,:]) 
            
            # evolve the Hamiltonian
            wg_power[idt,:] = self.evolve(length, ideal)
            
        
        return wg_power
    

    def generate_interaction_Hamiltonian(self, time_range, input_voltage, ideal = 0):
        """
        This method is to calculate the interaction Hamiltonian as a function of time.
        
        time_range   : an array of length num_points containing the time points where are we sampling the inputs and outputs
        input_voltage: an array of size (num_points,num_electrodes) storing the voltage at each time instant for each electrode. Each column represents the voltages applied over one electrode, with rows encoding time instants. 
        ideal        : a parameter that determines if we take the equiavalent circuit model into account for the simulation or not. The default value is 0. Use 0 for modeling the circuit effects, 1 for neglecting the circuit effects.     

        returns the interaction Hamiltonian as a function of time, in the form of an array of size (num_points,num_wg,num_wg).
        """
        
        # calculate the parameters of the transfer function of the LTI system modeling the chip behaviour
        num,den = self.ct_to_tf(self.circuit_params[0],self.circuit_params[1],self.circuit_params[2],self.circuit_params[3],self.circuit_params[4],self.circuit_params[5],self.circuit_params[6],self.circuit_params[7])
        
        # define the lti obejct 
        circuit = lti(num,den)
        
        # evaluate the circuit response of the input voltage 
        voltages_w_decay = np.array([ circuit.output(v,time_range)[1] for v in input_voltage.T])
        
        # initialize the array to store the Hamiltonian at each time instant        
        time_varying_Hamiltonian = np.zeros((time_range.size,self.num_wg,self.num_wg))
        
        # Get the zero-voltage hamiltonian
        self.set_voltages(np.zeros((self.num_electrodes)))
        self.set_hamiltonian()
        H0 = self.hamiltonian
        
        # loop over all time instants
        for idt,t in enumerate(time_range):
            
            # check if we want to model circuit non-idealities
            if ideal==0:
                
                # use circuit response
                self.set_voltages(voltages_w_decay[:,idt])
            
            else:
                
                # use ideal pulses
                self.set_voltages(input_voltage[idt,:])
            
            # evaluate the Hamiltonian at this point
            self.set_hamiltonian()
            
            # remove the zero-voltage component
            time_varying_Hamiltonian[idt,:] = self.hamiltonian - H0

        return time_varying_Hamiltonian
