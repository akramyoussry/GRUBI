"""
This module implements the machine learning-based model for the chip. It has three classes:
    Param_to_Ham_layer: This is an internal class for constructing Hamiltonians.
    Quantum_Evo_Layer : This is an internal class to implement quantum evolution and interferometer measurements calculations.
    Coupling_Layer    : This is an internal class to model coupling losses at the output.
    complexMLmodel    : This is the main class that defines machine learning model for the chip.  
"""

# Preamble
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,constraints,initializers,Model,backend
import pickle
import time
import zipfile    
import os
#####################################################################################################################    
###########################################################################################################
###########################################################################################################

class Param_to_Ham_Layer(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the Hamiltonian parameters as input, and generates the
    Hamiltonain matrix as an output at each time step for each example in the batch
    """
    
    def __init__(self, **kwargs):
        """
        Class constructor 
        """
        # this has to be called for any tensorflow custom layer
        super(Param_to_Ham_Layer, self).__init__(**kwargs)
    
    
    def build(self,input_shape):
        """
        This method must be defined for any custom layer, here you define the training parameters.
        
        input_shape: a tensor that automatically captures the dimensions of the input by tensorflow. 
        """    
        # get the the number of paramters
        self.num_wg = input_shape.as_list()[-1]

        # define the trainable parameters representing the zero-voltage Hamiltonian
        self.H0 = self.add_weight(name="H0", shape=tf.TensorShape((self.num_wg,self.num_wg)), initializer=initializers.RandomUniform(minval=0, maxval=100, seed=30),trainable=True)
               
        # define an operator to convert lower matrix into pure imaginary
        self.complex_operator = -1j * np.ones((self.num_wg,self.num_wg)) * np.tri(self.num_wg, self.num_wg, -1) + np.ones((self.num_wg,self.num_wg))*np.tri(self.num_wg,self.num_wg,0).T

        # this has to be called for any tensorflow custom layer
        super(Param_to_Ham_Layer, self).build(input_shape)

    
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
        
        # add zero voltage parameters to the input
        H = tf.cast( x + self.H0 , tf.complex64)

        # retreive the complex opertor 
        complex_operator = tf.constant( self.complex_operator, dtype=tf.complex64)
        
        # add two extra dimensions for batch and time
        complex_operator = tf.expand_dims(complex_operator, 0)
        complex_operator = tf.expand_dims(complex_operator, 0)
        
        # construct a tensor in the form of a row vector whose elements are [d1,d2,1,1], where d1 and d2 correspond to the
        # number of examples and number of time steps of the input
        temp_shape = tf.concat( [tf.shape(x)[0:2],tf.constant(np.array([1,1],dtype=np.int32))],0 )

        # repeat the input ket colmun along the batch and time dimensions
        complex_operator = tf.tile( complex_operator, temp_shape )
        
        # apply the complex operator to convert lower traingular part into pure imaginary
        H = tf.multiply(H, complex_operator)        
        
        # convert to symmetric matrix by doing H+H' [permute index 3 and 2]
        H = tf.add(H, tf.transpose(H,[0,1,3,2], conjugate=True) )
        
        return H
###############################################################################

class Quantum_Evo_Layer(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes Hamiltonian as input, and generates the ouptut probability
    distribution as output at each time step for each example in the batch
    """
    
    def __init__(self, length, initial_state, **kwargs):
        """
        Class constructor.
        
        length: The physical length of the chip that goes inside the evolution operator.
        initial_state: The column vector representing the inital state before evolution.
        """    
        
        # here we define the length including the imaginary unit, so we can later use it directly with the expm function
        self.length = length
        
        # specify the initial state
        self.initial_state = initial_state
        
        # we must call thus function for any tensorflow custom layer
        super(Quantum_Evo_Layer, self).__init__(**kwargs)
    
    def call(self, x):        
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: The tensor representing the input to the layer. This is passed automatically by tensorflow. 
        """            
        # evaluate -i*H*l
        Hamiltonian = tf.cast(x * self.length, tf.complex64) * -1j
        
        #evaluate U =expm(-i*H*l)
        U = tf.linalg.expm( Hamiltonian )
        
        # add an extra dimenstion to the tensor representing initial state, to represent time 
        psi_0 = tf.expand_dims(tf.constant(self.initial_state) ,0)
        
        # add another dimension to represent batch
        psi_0 = tf.expand_dims(psi_0,0)
        
        # construct a tensor in the form of a row vector whose elements are [d1,d2,1,1], where d1 and d2 correspond to the
        # number of examples and number of time steps of the input
        temp_shape = tf.concat( [tf.shape(x)[0:2],tf.constant(np.array([1,1],dtype=np.int32))],0 )
        
        # repeat the input ket colmun along the batch and time dimensions, and convert to complex64 datatype
        psi_0 = tf.tile( psi_0, temp_shape )
        psi_0 = tf.cast(psi_0, tf.complex64)

        # evalaue U \psi_0
        psi_t = tf.squeeze( tf.matmul(U,psi_0), -1 )
                
        # calculate the interferometer power distribution
        power_distribution =  tf.square( tf.abs(0.5*(1+psi_t)) ) 
        interferometer_distribution =  tf.square( tf.abs(0.5*(1j+psi_t)) ) 
        
        # concatentate the amplitudes and relative phases over each other
        output = tf.concat([power_distribution, interferometer_distribution],-1)
        
        return output
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'length': self.length,
            'initial_state': self.initial_state,
        })
        return config
###############################################################################
class Coupling_Layer(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the output waveguide
    power distribution as input and generates the normalized power distribution 
    after modeling coupling losses.
    """
    # class constructor
    def __init__(self, **kwargs):
        """
        Class constructor 
        """
        # we must call this function for any tensorflow custom layer
        super(Coupling_Layer, self).__init__(**kwargs)

    def build(self,input_shape):
        """
        This method must be defined for any custom layer, here you define the training parameters.
        
        input_shape: a tensor that automatically captures the dimensions of the input by tensorflow. 
        """        
        # retreive the number of waveguides
        self.num_wg = input_shape[-1]
        
        # define a list of trainable tensorflow parameters representing the coupling coefficients 
        coupling_coeff = []
        for idx_wg in range(self.num_wg):
            coupling_coeff.append(self.add_weight(name="e%s"%idx_wg, shape=tf.TensorShape(()), initializer=initializers.Constant(1.0), trainable=True, constraint = constraints.non_neg()) ) 
        
        # convert the list of tensors to one tensor
        coupling_coeff = tf.convert_to_tensor(coupling_coeff)
        
        # add an extra dimension to represent time 
        coupling_coeff = tf.expand_dims(coupling_coeff,0)
        
        # add another dimension to represent batch
        coupling_coeff = tf.expand_dims(coupling_coeff,0)
        
        # store this tensor as a class member so we can access it from othe methods in the class 
        self.coupling_coeff = coupling_coeff
        
        # this has to be called for any tensorflow custom layer
        super(Coupling_Layer, self).build(input_shape)
    
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: The tensor representing the input to the layer. This is passed automatically by tensorflow. 
        """    
        # construct a tensor in the form of a row vector whose elements are [d1,d2,1], where d1 and d2 correspond to the
        # number of examples and number of time steps of the input       
        temp_shape = tf.concat( [tf.shape(x)[0:2], tf.constant([1])],0 )
        
        # repeat the coupling coefficients along the batch and time dimensions
        coupling_coeff    = tf.tile(self.coupling_coeff, temp_shape)
        
        # multiply elemen-wise each power with its coupling coefficient
        power = tf.multiply(x, coupling_coeff)
        
        # normalize the result along the last dimension
        total_power = tf.reduce_sum(power, axis=-1, keepdims=True)
        power = tf.divide(power,total_power)
        return power
###############################################################################
class complexMLmodel():
    """
    This is the main class that defines machine learning model for the chip.
    """    
      
    def __init__(self, num_wg,  length):
        """
        Class constructor.
        
        num_wg: The nuumber of waveguides in the chip
        length: The physical length of the chip 
        """
        self.num_wg            = num_wg
        self.num_electrodes    = 2 * num_wg
        self.length            = length
        self.mode              = 0
        # a new tensorflow input layer
        input_voltages = layers.Input(shape=(None,self.num_electrodes), name="input_voltages")
        
        # add a GRU of 60 hidden nodes that is connected to the corresponding input layer of this electrode
        Parameters = layers.GRU(60, return_sequences=True)(input_voltages) 

        # add a normal neuron layer with linear activation that acts over each time slice 
        Parameters = layers.Dense(num_wg**2, activation='linear')(Parameters)

        # add a reshape layer to convert into a matrix
        Parameters = layers.Reshape((-1,num_wg,num_wg))(Parameters)
        
        # add the custom defined tensorflow layer that construct the Hamiltonian from parameters
        Hamiltonian = Param_to_Ham_Layer(trainable=False, name="Parameters")(Parameters)
        
        # define a list of input "basis" initial states 
        self.initial_states = [np.reshape( np.eye(self.num_wg)[:,idx_wg], (self.num_wg,1) ) for idx_wg in range(self.num_wg)]
        
        Distributions      = [ Quantum_Evo_Layer(length=length, initial_state=initial_state, name="Distributions_%d"%idx_initial_state)(Hamiltonian) for idx_initial_state,initial_state in enumerate(self.initial_states)]
         
        # define now the tensorflow model 
        self.model    = Model( inputs = input_voltages, outputs = Distributions )
        
        # specify the optimizer and loss function for training 
        self.model.compile(optimizer=optimizers.RMSprop(learning_rate=0.01), loss='mse')
        
        # print a summary of the model showing the layers, their connections, and the number of training parameters
        self.model.summary()

    def callibrate_zero_voltage(self, measurements):
        """
        This method is to train the zero-voltage parameters of the Hamiltonian, as well as the coupling losses coefficients. 
        
        measurements: A list of length equal to the number of initial basis states used in the model. Each item in the list is a numpy array that stores the measured power distribution when applying zero control voltages, with shape (1,1,number of waveguides)
        """    
        # define an input layer 
        input_params = layers.Input(shape=(None,self.num_wg**2))
        
        # add a reshape layer to convert into a matrix
        Parameters = layers.Reshape((-1,self.num_wg,self.num_wg))(input_params)

        # define the Hamiltian Layer with trainable zero-voltage paramaters
        Hamiltonian    = Param_to_Ham_Layer(trainable = True, name="Parameters")(Parameters)
        
        # define a list of evolution layers corresponding to each waveguide [learn one column of U=exp(iHt) at a time]
        Distributions   = [ Quantum_Evo_Layer(length=self.length, initial_state=initial_state, name="Distributions_%d"%idx_initial_state)(Hamiltonian) for idx_initial_state,initial_state in enumerate(self.initial_states)]
        
        # define a tensorflow model 
        model          = Model(inputs=input_params, outputs=Distributions)
        model.compile(optimizer=optimizers.RMSprop(learning_rate=0.01), loss='mse')
        model.summary()
        
        # do the training by providing with an input of zero voltages, and output is the measured power from the chip
        training_x = np.zeros((1,1,self.num_wg*self.num_wg))

        model.fit(training_x, measurements, epochs=100000, verbose=0)
        print("Actual measurements are: ")
        print(measurements)
        print("Predicted measurements are: ")
        print(model.predict(training_x))
        
        # save the trained model automatically as a temp file with the current time as filename
        tmp_file_name = "trained_model_%s.tmp.h5" % (time.strftime("%Y-%m-%d_%H-%M-%S"))
        model.save(tmp_file_name)
        
        # load the trainined parameters for all these layers
        self.model.load_weights(tmp_file_name, by_name=True)

        # remove the tmp file
        os.remove(tmp_file_name)
        
        return model.get_weights()
    
    def train_model(self, training_x, training_y, epochs, batch_size=100):
        """
        This method is for training the model given the training set
        
        training_x: A numpy array that stores the control voltages with shape (number of examples,number of time steps, number of electrodes).
        training_y: A list of numpy arrays of length equal to the number of bases used in the model (in the current implementation it is equal to number of waveguides). Each item in the list is a numpy array stroing the measured output power distribution with shape (number of examples,number of time steps, number of electrodes).
        epochs    : The number of iterations to do the training     
        """        
        # retreive the batch size from the training dataset
        num_examples  = training_x.shape[0]
        
        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        self.training_history = self.model.fit(training_x, training_y, epochs=epochs, batch_size=batch_size,verbose=2).history["loss"]
        
       
    def train_model_val(self, training_x, training_y, val_x, val_y, epochs, batch_size=100):
        """
        This method is for training the model given the training set and the validation set
        
        training_x: A numpy array that stores the time-domain represenation of control pulses of dimensions (number of examples, number of time steps, 1, number of axes)
        training_y: A numpy array that stores the measurement outcomes (number of examples, number of measurements).
        val_x     : The validation input array [similar to training_x]
        val_y     : The valiation desired output array [similar to training_y]  
        epochs    : The number of iterations to do the training     
        batch_size: The batch size    
        """        

        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        h  =  self.model.fit(training_x, training_y, epochs=epochs, batch_size=batch_size,verbose=2,validation_data = (val_x, val_y)) 
        self.training_history  = h.history["loss"]
        self.val_history       = h.history["val_loss"]
    
    def predict_measured_wg_power(self, testing_x, idx_initial_state=0):
        """
        This method is for predicting the measured output waveguide power distribution using the model. Usually called after traning.
        
        testing_x:         A numpy array that stores the control voltages with shape (number of examples,number of time steps, number of electrodes).
        idx_initial_state: The index of which distribution do we want to obtain from the list of possible initial basis states used in the model.
        """        
        return self.model.predict(testing_x)[idx_initial_state]
    
    def predict_wg_power(self, testing_x, idx_initial_state=0):
        """
        This method is for predicting the actual output waveguide power distribution (i.e. without coupling losses), using the model. Usually called after traning.
        
        testing_x:         A numpy array that stores the control voltages with shape (number of examples,number of time steps, number of electrodes).
        idx_initial_state: The index of which distribution do we want to obtain from the list of possible initial basis states used in the model.
        """            

        return self.model.predict(testing_x)[idx_initial_state]
    
    def predict_Hamiltonian(self,testing_x):
        """
        This method is for predicting the Hamiltonian using the model. Usually called after training.
        
        testing_x: A numpy array that stores the control voltages with shape (number of examples,number of time steps, number of electrodes).
        """
        
        # define a new model that connects the input voltage and the GRU output 
        chip_voltage_model = Model(inputs=self.model.input, outputs=self.model.get_layer('Parameters').output)
    
        # evaluate the output of this model
        return chip_voltage_model.predict(testing_x)
      
    def predict_zero_voltage_Hamiltonian(self):
        """
        This method is to evaulate the zero-voltage Hamiltonian H0. Usually called after callibrating the zero-voltage parameters   
        """

        # define an input layer
        input_params = layers.Input(shape=(None,self.num_wg,self.num_wg))
        
        # define a model that to capture only H0
        H0_model = Model(inputs = input_params, outputs = Param_to_Ham_Layer(name="Parameters")(input_params) )
        
        # retreive the weights from the full model
        weights = self.model.get_layer("Parameters").get_weights()
        
        # copy these weights to the new small model
        H0_model.get_layer("Parameters").set_weights(weights)
         
        # set interaction Hamiltonain to be zero
        H1 = np.zeros( (1,1,self.num_wg,self.num_wg) )

        # predict full Hamiltonian H = H0 + H1 = H0
        H0 = H0_model.predict(H1)
        
        # return as n-by-n matrix
        return H0
    
    def predict_interaction_Hamiltonian(self,testing_x):    
        """
        This method is to evaulate the interaction Hamiltonian H1. Usually called after training.   
        
        testing_x: A numpy array that stores the control voltages with shape (number of examples,number of time steps, number of electrodes).
        """        
        # evaluate full Hamiltonian
        H = self.predict_Hamiltonian(testing_x)
        
        # evaluate zero-voltage Hamitlonian
        H0 = self.predict_zero_voltage_Hamiltonian()
        
        # return the difference
        return H-H0          
    

    def scaled_tanh(self,x):
        """            
        This is intenal method to implement a scaled tanh to be used as an activation function for the controller.
        
        x: input real number.
        """
        vmax = 10.0
        return backend.tanh(x)*vmax*0.5
    
    def construct_controller(self, mode=2):
        """
        This method is to construct the machine learning based-model for the controller. The functionality of the controller is to obtain the control voltages needed to acheive some unitary gate, undoing any distortions introduced by the parasitics of the chip. 
        mode: whether input is real Hamiltonian-->0 , complex Hamiltonian-->1, or complex unitary-->2, default 0
        """        
        
        # define input layer for the target unitary
        self.mode = mode
        if mode==0:
            input_matrix = layers.Input( shape=(None,self.num_wg*(self.num_wg+1)//2) )
        elif mode==1:
            input_matrix = layers.Input( shape=(None,self.num_wg*(self.num_wg+1)) )
        else:
            input_matrix = layers.Input( shape=(None,self.num_wg*self.num_wg*2) )

        # add a dummy layer that just generates zeros to model the electrodes that are grounded
        zero_output      = layers.Lambda(lambda x: x[:,:,0:1]*0, output_shape=(None,None,1))(input_matrix)     
                        
        # add an GRU of 60 hidden nodes that is connected to the corresponding input layer of this electrode
        control_voltages = layers.GRU(60, return_sequences=True)(input_matrix)
        
        # add a neuron layer with the scaled tanh activation that acts over each time slice
        control_voltages = layers.Dense(self.num_electrodes-2,activation=self.scaled_tanh)(control_voltages)

        # concatenate with the dummy zero voltages layer 
        control_voltages = layers.Concatenate()([zero_output, control_voltages,zero_output])
        
        # define a tensorflow model that captures the trainined GRU part of the chip model without the quantum part
        Hamiltonian_model    = Model(inputs=self.model.input, outputs=self.model.outputs, name='Hamiltonian_model')
        
        # prevent training any layer of the alread-trainined GRU
        for layer in Hamiltonian_model.layers:
            layer.trainable  = False
        
        # define the controller model by connecting the new GRU with the pretrained GRU
        controller_train_model = Model(inputs = input_matrix, outputs = Hamiltonian_model(control_voltages) )
                
        # specify the optimizer and loss function for training
        controller_train_model.compile(optimizer=optimizers.RMSprop(), loss='mse', metrics=['mse'])
        
        # print a summary of the model showing the layers, their connections, and the number of training parameters
        controller_train_model.summary()
        
        # store the full control part [inverse model+model] as a class member 
        self.controller_train_model = controller_train_model
        
    def flatten_input(self, H):
        """
        This is an internal method for reshaping input matrices to the GRU layer of the controller.
        
        H: a numpy array representing the target matrx input (target unitary) with shape (number of examples,number of time steps, number of waveguides, number of waveguides)
        """        
        # retreive the batch size from the training dataset
        num_examples = H.shape[0]
        num_points   = H.shape[1]
        
        # initialize the array for expanding the upper triangular part
        params_r = np.zeros( (num_examples,num_points,self.num_wg*(self.num_wg+1)//2) )
        params_i = np.zeros( (num_examples,num_points,self.num_wg*(self.num_wg+1)//2) )
        
        # define an anynomus function to extract upper triangular part of a matrix
        get_upper = (lambda x: x[np.mask_indices(self.num_wg, np.triu)])
        
        if self.mode==0 or self.mode==1:
            #extract upper triangular part of the Hamiltonian
            for idx_ex in range(num_examples):
                for idx_t in range(num_points):
                    params_r[idx_ex,idx_t,:] = np.real(get_upper(H[idx_ex,idx_t,:]))
                    params_i[idx_ex,idx_t,:] = np.imag(get_upper(H[idx_ex,idx_t,:]))
        if self.mode==0:
            return params_r
        elif self.mode==1:
            return np.concatenate((params_r, params_i),axis=-1)
        else:
            return np.concatenate((np.real(np.reshape(H,(num_examples, num_points, self.num_wg*self.num_wg))), np.imag(np.reshape(H,(num_examples, num_points, self.num_wg*self.num_wg))) ),axis=-1)

    def train_controller(self, target_H, target_y, epochs):
        """
        This is the method to train the controller to find the set of control voltages to obtain the target power distribution
        
        target_H: a numpy representing the sequence of targets of shape (number of examples,number of time steps, number of waveguides, number of waveguides) 
        target_y: A list of numpy arrays of length equal to the number of bases used in the model (in the current implementation it is equal to number of waveguides). Each item in the list is a numpy array stroing the ideal output power distribution (i.e.without coupling losses with shape) of shape (number of examples,number of time steps, number of electrodes).
        """
        # retreive the batch size from the training dataset
        num_examples = target_H.shape[0]
        
        #flatten out the input
        params = self.flatten_input(target_H)
        
        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        self.control_training_history = self.controller_train_model.fit(params, target_y, epochs=epochs, batch_size=num_examples,verbose=2).history 
        
    def predict_control_voltage(self,target_H):
        """
        This method predicts the control voltage, usually called after training contoller.
        
        target_H: a numpy representing the sequence of target Hamiltonians of shape (number of examples,number of time steps, number of waveguides, number of waveguides) 
        """
        #flatten out the input
        params = self.flatten_input(target_H)
        
        # Define a model to capture the control part only [inverse model witout model -qunatum] 
        controller_model = Model(inputs = self.controller_train_model.input, outputs = self.controller_train_model.layers[-2].output )
    
        # evaluate the output of the controller model [inverse model-model-quantum]
        return controller_model.predict(params)

    def predict_controlled_wg_power(self,target_H, input_waveguide=0):
        """
        This method predicts the waveguide power distribution if we apply the control voltage.
        
        target_H: a numpy representing the sequence of target Hamiltonians of shape (number of examples,number of time steps, number of waveguides, number of waveguides) 
        """        
        # evaluate control voltage [i.e. the result of the inverse model]
        control_voltage     = self.predict_control_voltage(target_H)
        
        # simulate the chip with the control voltage
        controlled_wg_power = self.predict_wg_power(control_voltage, input_waveguide)

        return controlled_wg_power
       
    def save_model(self, filename):
        """
        This method is to export the model to an external .mlmodel file
        
        filename: The name of the file (without any extensions) that stores the model.
        """
        
        # first save the ml models
        self.model.save(filename+"_model.h5")
        self.controller_train_model.save(filename+"_controller.h5")
        
        # second, save all other variables
        data = {'training_history':self.training_history, 
                'val_history'     :self.val_history,
                'control_training_history':self.control_training_history, 
                'num_wg':self.num_wg,
                'num_electrodes':self.num_electrodes,
                'length':self.length,
                'mode':self.mode
                }
        f = open(filename+"_class.h5", 'wb')
        pickle.dump(data, f, -1)
        f.close()
	
        # zip everything into one zip file
        f = zipfile.ZipFile(filename+".mlmodel", mode='w')
        f.write(filename+"_model.h5")
        f.write(filename+"_controller.h5")
        f.write(filename+"_class.h5")
        f.close()
        
        # now delete all the tmp files
        os.remove(filename+"_model.h5")
        os.remove(filename+"_controller.h5")
        os.remove(filename+"_class.h5")

    def load_model(self, filename):
        """
        This method is to import the models from an external .mlmodel file
        
        filename: The name of the file (without any extensions) that stores the model.
        """       
        #unzip the zipfile
        f = zipfile.ZipFile(filename+".mlmodel", mode='r')
        f.extractall()
        f.close()
        
        # first load the class variables
        f = open(filename+"_class.h5", 'rb')
        data = pickle.load(f)
        f.close()          
        
        self.training_history          = data['training_history']
        self.val_history               = data['val_history']
        self.control_training_history  = data['control_training_history']
        self.num_wg                    = data['num_wg']
        self.num_electrodes            = data['num_electrodes']  
        self.length                    = data['length']
        self.mode                      = data['mode']
        
        # second, load ml model
        self.model.load_weights(filename+"_model.h5")       

        # sthird, construct the controller and load its weights
        self.construct_controller(self.mode)
        self.controller_train_model.load_weights(filename+"_controller.h5")
        
        # now delete all the tmp files
        os.remove(filename+"_model.h5")
        os.remove(filename+"_controller.h5")
        os.remove(filename+"_class.h5")
