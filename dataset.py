"""
This module contains two functions for creating and loading dataset of examples for training and testing the MLmodel class implemented in mlmodel.py
"""

# Preamble
import numpy as np

def create_dataset(filename, description, num_examples, rwa_simulator, time_range, seed=30, verbose=0):
    """
    This function creates a dataset of pulse waveforms with random amplitudes,
    durations and start times, and corresponding waveguide power distribution in 
    the ideal and drifting cases, as well as the ideal Hamiltonians. It needs to be
    passed an instance of the class (FourElectrodeArraySimulator).
    
    filename     : The name of the file to store the datatset.
    description  : A string that user can use to store a description of the datatset.
    num_examples : A list of two elements that represent number of training examples and testing examples.
    rwa_simulator: An instance of the "FourElectrodeArraySimulator" that is already constructed with the required parameters to model the chip.
    time_range   : A numpy array that stores time interval. 
    seed         : A number to initialize the random number generator. Use the same seed to generate the same examples.
    verbose      : Set to 1 to display a message reporting the progress of the generation, 0 to disable.  
    """   
    
    # initialize the random number generator at some known point for reproducability
    np.random.seed(seed) 
    
    # extract the number of examples for each of the training and testing sets
    num_training_examples,num_testing_examples = num_examples
    total_num_examples = num_training_examples + num_testing_examples
    
    # retreive some parameters from the simulator
    num_electrodes = rwa_simulator.num_electrodes
    num_wg         = rwa_simulator.num_wg
    length         = rwa_simulator.length
    
    # retreive some parameters from about time domain
    num_points     = time_range.size 
    start_time     = time_range[0]
    end_time       = time_range[-1]
    ts             = time_range[1]-time_range[0]
    
    # initialze the arrays for the dataset. They should be three dimensional as required by tensorflow.
    training_x       = np.zeros((total_num_examples,num_points,num_electrodes))  # input voltages 
    training_y       = [np.zeros((total_num_examples,num_points,num_wg)) for idx_wg in range(num_wg)] # output power distribution
  
    # the dataset consists of pulses of different amplitudes, widths, and start times
    for idx_example in range(total_num_examples):
        
        # print a progress message if we want
        if verbose==1:
            print("Generating Example %d of %d"%(idx_example+1,total_num_examples), end='\r')
            
        #initialize voltages for this example
        voltages = np.zeros((num_points,num_electrodes))
        
        # select randomly some electrodes to apply pulses, such that we fix the first and last to ground
        
        #generate random number representing the indices
        active_electrodes = np.random.randint(1, 2**(num_electrodes - 2) )
        
        # represent the number in binary to retreive the indices of electrodes to which we shall apply the pulses
        active_electrodes = np.binary_repr(active_electrodes, num_electrodes-2)

        # define the support of the pulses for this example randomly
        Tstart   = np.random.uniform(low = start_time, high = (start_time + end_time)*0.5)
        Twidth   = np.random.uniform(low = 10*ts, high = (start_time + end_time)*0.5)
        Tend     = Tstart + Twidth
        
        # loop over alll indices and check which one to apply a pulse
        for idx_electrode in range(num_electrodes-2):
            if active_electrodes[idx_electrode]=='1':
                # define ampitude of the input pulse for this active electrode randomly
                A        = np.random.uniform(low = -5, high = 5)

                # apply the pulse to this electrode
                voltages[:,idx_electrode+1] = A*(time_range>Tstart)*(time_range<Tend)
        
        # store this example
        training_x[idx_example,:] = voltages
        
        # simulate the chip to evaluate the ouput power distribution and the chip response
        for idx_wg in range(num_wg):
            rwa_simulator.set_input_waveguide(idx_wg)
            training_y[idx_wg][idx_example,:]       = rwa_simulator.evolve_varying_voltage(time_range,voltages,length, ideal = 0)

    # split the datsaet into training and testing datasets
    testing_x       = training_x[num_training_examples:,:,:]
    testing_y       = [training_y[idx_wg][num_training_examples:,:,:] for idx_wg in range(num_wg)]  

    training_x       = training_x[0:num_training_examples,:,:]
    training_y       = [training_y[idx_wg][0:num_training_examples,:,:] for idx_wg in range(num_wg)]

    # define a list of waveguide power distributions corresponding to zero-voltage input
    zero_voltage_measurements = []
    
    # set the input voltages to zero
    rwa_simulator.set_voltages(np.zeros((num_electrodes,1)))
    
    # loop over all possible waveguides
    for idx_wg in range(num_wg):
    
        # light the laser into this waveguide
        rwa_simulator.set_input_waveguide(idx_wg)
    
        # evolve with zero voltage, and append to the list
        zero_voltage_measurements.append( np.reshape(rwa_simulator.evolve(rwa_simulator.length), (1,1,num_wg) ) )

    # store the dataset in an external file
    np.savez(filename, description=description, time_range = time_range, training_x = training_x, training_y = training_y,
             testing_x  = testing_x, testing_y  = testing_y, zero_voltage_measurements = zero_voltage_measurements)
###############################################################################
def load_dataset(filename):    
    """
    This function loads a datset file created by the create_dataset function
    """    
    # load the dataset from an external file
    dataset = np.load(filename)
    
    training_x       = dataset['training_x']
    training_y       = list(dataset['training_y'])

    testing_x        = dataset['testing_x']
    testing_y        = list(dataset['testing_y'])

    time_range       = dataset['time_range']
    
    zero_voltage_measurements = list(dataset['zero_voltage_measurements'])    

    dataset.close()
    
    return training_x,training_y, testing_x, testing_y, zero_voltage_measurements, time_range
