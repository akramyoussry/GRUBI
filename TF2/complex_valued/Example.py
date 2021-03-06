#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm,expm,block_diag
from itertools import accumulate
from simulator import FourElectrodeArraySimulator
from dataset import create_dataset,load_dataset
from complexmlmodel import complexMLmodel


# <H2> Step 1: Construct the training dataset </H2>

# In[ ]:


# define the simulation parameters
num_wg     = 3
length     = 3.6e-2
n0         = 2.1753
delta_n    = 5e-6                
c0         = 100.
delta_c1   = 1.5
delta_c2   = -1.3 
wavelength = 808.e-9

circuit_params = [70, 70, 700, 700, 30e3, 100e3, 10e-6, 25e-6]
coupling_coeff = [0.9,0.8,0.5]

# define the time domain to be from t=0 to t=500 milliseconds
start_time = 0
end_time   = 200
Ts         = 0.2
time_range = np.arange(start_time,end_time,Ts)*(1e-3)
num_points = time_range.size

# define the simulator which will generate the traning set
rwa_simulator = FourElectrodeArraySimulator(num_wg, coupling_coeff=coupling_coeff, n0=n0, delta_n=delta_n, c0=c0,
                  delta_c1=delta_c1, delta_c2=delta_c2, wavelength=wavelength, circuit_params=circuit_params)

# define the dataset information
dataset_filename      = "dataset_complex.npz"
num_training_examples = 3500
num_testing_examples  = 500
batch_size            = 500 
num_examples          = [num_training_examples,num_testing_examples]
seed                  = 30 # for reproducability
description           = "set of %d training and %d testing pulses of random amplitudes, widths, and shifts, with seed=%d, circuit parameters=%s, and  coupling coefficients= %s. Pulses across different electrodes start and end at the same time."%(num_training_examples,num_testing_examples,seed,str(circuit_params), str(coupling_coeff)) 


# In[ ]:


# create a new dataset and save it externally [verbose = 1 to show the progress of the generation]
# SKIP THIS STEP IF ALREADY GENERATED THE DATASET
#create_dataset(dataset_filename, description, num_examples, rwa_simulator, time_range, seed, verbose = 1)


# In[ ]:


# load the created dataset
training_x,training_y, testing_x, testing_y,  zero_voltage_measurements, time_range = load_dataset(dataset_filename)


# <H2> Step 2: Train the machine learning model to learn the dynamics of the chip </H2>

# In[ ]:


# a) define the model
model = complexMLmodel(num_wg, length)
model_filename = "model_7_12_2020_complex" # for importing/exporting the model externally


# In[ ]:


# b) calibrate the zero-voltage parameters & coupling coefficients (Skip steps 2b, 2c if already trained and loading from file)    
zero_voltage_parameters = model.callibrate_zero_voltage(zero_voltage_measurements)


# In[ ]:


# c) train the model with the training set
num_iter = 10000
model.train_model_val(training_x, training_y, testing_x, testing_y, num_iter, batch_size=batch_size)


# In[ ]:


# or if we already trained it and saved the file, then we can load the trained model
model.load_model(model_filename)


# In[ ]:


# d) plot the training history
plt.figure(figsize=[4.8, 3.8])
plt.loglog(model.training_history)
plt.loglog(model.val_history)
plt.legend(['Training', 'Testing'], fontsize=11)
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('MSE',fontsize=11)
plt.xscale('log')
plt.xticks(sum([[i*j for i in range(1,11)] for j in [1,10,100,1000]],[])+[20000,30000],fontsize=11)
plt.yscale('log')
plt.yticks(sum([[i*j for i in range(1,11)] for j in [0.0001, 0.001,0.01]],[]), fontsize=11)
plt.grid(True, which="both")
plt.savefig('model_complex_training.pdf',format='pdf', bbox_inches='tight')


# In[ ]:


# display MSE for the training dataset
print("%e"%model.training_history['loss'][-1])


# In[ ]:


# display MSE for the testing dataset
print( "%e"%model.model.evaluate(testing_x, testing_y, batch_size=num_testing_examples)[0] )


# In[ ]:


# display the zero-voltage Hamiltonian
print(model.predict_zero_voltage_Hamiltonian())
rwa_simulator.set_voltages(np.zeros((num_wg*2)))
rwa_simulator.set_hamiltonian()
print(rwa_simulator.hamiltonian)


# In[ ]:


# e) make a user interactive cell to view the results on the testing dataset 

# use the model to predict power distribution and chip response from the training dataset
predicted_wg_power_training     = [model.predict_measured_wg_power(training_x, idx_wg) for idx_wg in range(num_wg)]

# use the model to predict power distribution and chip response from the testing dataset
predicted_wg_power_testing      = [model.predict_measured_wg_power(testing_x,  idx_wg) for idx_wg in range(num_wg)]


# In[ ]:


# define a function to display a particular example      
def update_display(Dataset, input_waveguide, idx_example):
    if(Dataset=="Training"):
        plt1 = training_x[idx_example,:]
        plt2 = training_y[input_waveguide][idx_example,:]
        plt3 = predicted_wg_power_training[input_waveguide][idx_example,:]
    else:
        plt1 = testing_x[idx_example,:]
        plt2 = testing_y[input_waveguide][idx_example,:]
        plt3 = predicted_wg_power_testing[input_waveguide][idx_example,:] 
    
    plt.figure(figsize=[15, 3.8*num_wg])
    for idx_wg in range(num_wg):            
        # plot the potential difference across each waveguide
        plt.subplot(num_wg,1,idx_wg +1)       
        plt.plot(time_range, plt1[:,idx_wg*2],'r',   label = "$V_%d(t)$"%(idx_wg*2))
        plt.plot(time_range, plt1[:,idx_wg*2 +1],'k',label = "$V_%d(t)$"%(idx_wg*2 +1))
        plt.xlabel('t')
        plt.ylabel("$V(t)$")
        plt.ylim([-5.1,5.1])
        plt.grid()
        plt.legend(ncol=2)
        plt.tight_layout()
     
    plt.figure(figsize=[8, 3*num_wg])
    for idx_wg in range(num_wg):
        plt.subplot(num_wg,2,idx_wg*2 +2)
        plt.plot(time_range,plt2[:,idx_wg+num_wg], label = "actual")
        plt.plot(time_range,plt3[:,idx_wg+num_wg], label = "predicted")
        plt.xlabel('t',fontsize=11)
        plt.ylabel(r"$P_%d\left(\theta=\frac{\pi}{2}, t\right)$"%idx_wg,fontsize=11)
        plt.xticks(np.arange(start_time,end_time+50,50)*1e-3,fontsize=11)
        plt.yticks(np.linspace(0,1,5),fontsize=11)
        plt.grid(True, which="both")
        plt.legend(fontsize=11)
            
        # plot the potential difference across each waveguide
        plt.subplot(num_wg,2,idx_wg*2 +1)
        plt.plot(time_range,plt2[:,idx_wg], label = "actual")
        plt.plot(time_range,plt3[:,idx_wg], label = "predicted")
        plt.xlabel('t',fontsize=11)
        plt.ylabel(r"$P_%d(\theta=0, t)$"%idx_wg,fontsize=11)
        plt.xticks(np.arange(start_time,end_time+50,50)*1e-3,fontsize=11)
        plt.yticks(np.linspace(0,1,5),fontsize=11)
        plt.grid(True, which="both")
        plt.legend(fontsize=11)
        plt.tight_layout()
    plt.savefig('ex_complex_%d_%d.pdf'%(idx_example, input_waveguide),format='pdf', bbox_inches='tight')
    
# add a widget for selecting the example

# widgets.interact(update_display, Dataset={'Testing','Training'}, input_waveguide=widgets.Dropdown(options=[idx_wg for idx_wg in range(num_wg)]), idx_example=widgets.IntSlider(min=0,max=testing_x.shape[0]-1,step=1, continuous_update=False) )

# or display directly
update_display("Testing",0,0)
update_display("Testing",2,381)


# In[ ]:


# e) make a user interactive cell to view the results on the testing dataset 

def construct_state(P):
    state   = np.zeros((P.shape[0],P.shape[1]//2),dtype=np.complex)
    for idx_t in range(P.shape[0]):
        P_r = P[idx_t, 0:num_wg]
        P_i = P[idx_t, num_wg:]
        mag2 = 2*(P_r + P_i) - np.sqrt(-4*P_r*P_r - 4*P_i*P_i + 4*P_r + 4*P_i+ 8*P_r*P_i - 1)
        state[idx_t,:] = 2*(P_r + 1j*P_i) - (0.5+0.5j)*(mag2+1)
    return state

def fidelity(phi,psi):
    return np.array([np.abs(phi[t,:].conj() @ psi[t,:])**2 for t in range(phi.shape[0])])

def construct_unitary(P):
    U = np.zeros((P[0].shape[0],num_wg,num_wg), dtype=np.complex)
    for t in range(P[0].shape[0]):
        for idx in range(num_wg):
            U[t,idx,:] = P[idx][t,:]
    return U
    
def gate_fidelity(U,V):
    return np.array([(np.abs(np.trace(U[t,:].conj().T @ V[t,:]))**2)/(num_wg**2) for t in range(U.shape[0])])

# define a function to display a particular example      
def update_display_2(Dataset, idx_example):
    phi=[]
    psi=[]
    plt.figure(figsize=[4.8,3.8])
    
    for idx_input_waveguide in range(num_wg):
        
        if(Dataset=="Training"):
            plt1 = training_x[idx_example,:]
            psi.append( construct_state(training_y[idx_input_waveguide][idx_example,:]) )
            phi.append( construct_state(predicted_wg_power_training[idx_input_waveguide][idx_example,:]) ) 

        else:
            plt1 = testing_x[idx_example,:]
            psi.append( construct_state(testing_y[idx_input_waveguide][idx_example,:]) )
            phi.append( construct_state(predicted_wg_power_testing[idx_input_waveguide][idx_example,:]) )
    
    phi = construct_unitary(phi)
    psi = construct_unitary(psi)
    plt2 = 1-gate_fidelity(phi,psi)
    
    plt.semilogy(time_range,plt2)
    plt.xlabel('t (sec)',fontsize=11)
    plt.ylabel('$1-F(t)$',fontsize=11)
    plt.xticks(np.arange(start_time,end_time+50,50)*1e-3,fontsize=11)
    plt.yscale('log')
    plt.yticks(sum([[i*j for i in range(1,11)] for j in [1e-5,1e-4,1e-3]],[]), fontsize=11)
    plt.grid(True, which="both")   
    plt.tight_layout()
    plt.savefig('infidelity_complex_%d.pdf'%idx_example,format='pdf', bbox_inches='tight')
    
# add a widget for selecting the example
# widgets.interact(update_display_2, Dataset={'Testing','Training'}, idx_example=widgets.IntSlider(min=0,max=testing_x.shape[0]-1,step=1, continuous_update=False) )

# or display directly
update_display_2("Testing",381)


# <H2> Step 3: Estimate the control voltage to achieve some standard gates </H2>

# In[ ]:


# Define some quantum gates we might want to implement

# Identity gate: [removes intrinsic coupling between waveguides]
U_Identity      = np.eye(3)
# Permutation gate [swap waveguide 1 and 2] 
U_Permutation12 = np.array([[0,1,0],[1,0,0],[0,0,1]]) 
# Permutation gate [swap waveguide 2 and 3] 
U_Permutation23 = np.array([[1,0,0],[0,0,1],[0,1,0]])
# Permutation gate [swap waveguide 1 and 3] 
U_Permutation13 = np.array([[0,0,1],[0,1,0],[1,0,0]])
# Hadamard gate [split power between waveguide 1 and 2] 
U_Hadamard_12   = np.array([[1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),-1/np.sqrt(2),0],[0,0,1]])
# Hadamard gate [split power between waveguide 2 and 3] 
U_Hadamard_23   = np.array([[1,0,0],[0,1/np.sqrt(2),1/np.sqrt(2)],[0,1/np.sqrt(2),-1/np.sqrt(2)]])
# Hadamard gate [split power between waveguide 1 and 3] 
U_Hadamard_13   = np.array([[1/np.sqrt(2),0,1j/np.sqrt(2)],[0,1,0],[1/np.sqrt(2),0,-1j/np.sqrt(2)]])
# Rotation pi/4 about x between 1 and 3
U_X_pi_4        = expm(-1j*0.25*np.pi*U_Permutation13)
#U_X_pi_4        = expm(-1j*0.1*U_Permutation13)
# Z gate between 1 and 3
U_Z13           = np.array([[1,0,0],[0,1,0],[0,0,-1]])
# Rotation pi/4 about z between 1 and 3
U_Z_pi_4        = expm(-1j*0.25*np.pi*U_Z13)
# Rotation pi/4 about z between 1 and 3
U_Z_pi_2        = expm(-1j*0.5*np.pi*U_Z13)
# Rotation pi/4 about z between 1 and 3
U_Z_pi          = expm(-1j*np.pi*U_Z13)
# Rotation 0.1 about z between q and 3 
U_HP            = expm(-1j*0.1*U_Z13)

# group all gates in one list
Gate_List = [U_Identity, U_Permutation12, U_Permutation23, U_Permutation13, U_Hadamard_12, U_Hadamard_23, U_Hadamard_13, U_X_pi_4, U_Z13, U_Z_pi_4, U_Z_pi_2, U_Z_pi, U_HP]
print(*Gate_List,sep='\n')

# calculate the corresponding Hamiltonians and group them in a list
Hamiltonian_List = [np.real_if_close(1j*logm(x))/length for x in Gate_List]
print(*Hamiltonian_List,sep='\n')


# In[ ]:


# define the time domain to be from t=0 to t=500 milliseconds
start_time = 0
end_time   = 280
Ts         = 0.2
time_range_control = np.arange(start_time,end_time,Ts)*(1e-3)
num_points = time_range_control.size

# define the desired switch sequence
sequence = [( [0,     50e-3], 0, "$I$"),                          # identity
            ( [50e-3, 90e-3], 3, "$X_{13}$"),                     # permutation13
            ( [90e-3, 130e-3],0, "$I$"),                          # identity
            ( [130e-3,170e-3],7, r"$R_{X_{13}}(\frac{\pi}{4})$"), # Rot X pi/4
            ( [170e-3,210e-3],0, "$I$"),                          # identity
            ( [210e-3,250e-3],12,r"$R_{Z_{13}}(0.1)$"),           # Rot Z 0.1
            ( [250e-3,280e-3],0, "$I$"),                          # identity
            #( [250e-3,280e-3],11),# Rot Z pi
            #( [280e-3,300e-3],0)  # identity
           ]
# construct Hamiltonian sequence
t  = np.reshape(time_range_control, (1,num_points,1,1))
Hamiltonian_sequence = sum( [np.kron( (t>=transition[0][0])*(t<transition[0][1]), Gate_List[transition[1]] ) for transition in sequence] )

# construct power distribution sequence
t  = np.reshape(time_range_control, (1,num_points,1))
powers_sequence = [sum([np.kron( (t>=transition[0][0])*(t<transition[0][1]), np.concatenate((np.abs(0.5*(Gate_List[transition[1]][:,idx_wg]+1))**2,np.abs(0.5*(Gate_List[transition[1]][:,idx_wg]+1j))**2),-1)  ) for transition in sequence]) for idx_wg in range(num_wg)]


# In[ ]:


# a) construct the control model (skip steps 3a,3b,3c if we did the training before and just importing the model file)
mode=2
model.construct_controller(mode)


# In[ ]:


# b) do the training 
num_iterations = 2000
model.train_controller(Hamiltonian_sequence, powers_sequence, num_iterations)


# In[ ]:


# c) save the whole model
model.save_model(model_filename)


# In[ ]:


# d) plot the training history
plt.figure(figsize=[4.8, 3.8])
plt.loglog(model.control_training_history['loss'])
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('MSE',fontsize=11)
plt.xscale('log')
plt.xticks(sum([[i*j for i in range(1,11)] for j in [1,10,100]],[])+[2000],fontsize=11)
plt.yscale('log')
plt.yticks(sum([[i*j for i in range(1,11)] for j in [0.01]],[]), fontsize=11)
plt.grid(True, which="both")
plt.savefig('controller_complex_training.pdf',format='pdf', bbox_inches='tight')


# In[ ]:


# e) plot the controlled waveguide power distribution
for idx_wg in range(num_wg): # loop over input waveguide
    plt.figure(figsize=[7.5, 3.5*num_wg])
    test_power = model.predict_controlled_wg_power(Hamiltonian_sequence, idx_wg)
    for idx_wg2 in range(2*num_wg): # loop over the distribution
        plt.subplot(2*num_wg,1,idx_wg2+1)
        plt.plot(time_range_control, powers_sequence[idx_wg][0,:][:,idx_wg2], label='ideal')
        plt.plot(time_range_control, test_power[0,:,idx_wg2], '--', label='controlled')
        plt.ylim([-0.1,1.1])
        plt.legend(ncol=2)
        plt.grid()
    plt.show()
    plt.savefig('controller_powers_%d.pdf'%idx_wg, format='pdf', bbox_inches='tight')    


# In[ ]:


# e) plot the controlled output state infedility
phi=[]
psi=[]
plt.figure(figsize=[10, 3.8])
for idx_wg in range(num_wg): # loop over input waveguide
    test_power = model.predict_controlled_wg_power(Hamiltonian_sequence, idx_wg)
    phi.append( construct_state(test_power[0,:]) )
    psi.append( construct_state(powers_sequence[idx_wg][0,:]) )
phi = construct_unitary(phi)
psi = construct_unitary(psi)
fid = 1-gate_fidelity(phi,psi)
plt.semilogy(time_range_control, fid)
plt.xlabel('t (sec)',fontsize=11)
plt.ylabel('1-F(t)', fontsize=11)
plt.grid(True, which="both")
plt.xticks([transition[0][0]for transition in sequence]+[sequence[-1][0][1]], fontsize=11)
plt.yticks(fontsize=11)
for transition in sequence:
    plt.arrow(transition[0][0],1.4e-1,transition[0][1]-transition[0][0],0, length_includes_head=True,head_width=0.025, head_length=2.5e-3,color="red")
    plt.arrow(transition[0][1],1.4e-1,transition[0][0]-transition[0][1],0, length_includes_head=True,head_width=0.025, head_length=2.5e-3,color="red")
    plt.text(0.5*(transition[0][0] + transition[0][1]), 1.4e-1, transition[2], fontsize=11,horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white',edgecolor='red'))
plt.savefig('controller_complex_infidelity.pdf',format='pdf', bbox_inches='tight')


# In[ ]:


# f) plot the control voltage
control_voltage = model.predict_control_voltage(Hamiltonian_sequence)[0,:]
plt.figure(figsize=[9, 4*num_wg])
for idx_electrode in range(num_wg*2):
    # plot the potential difference across each waveguide
    plt.subplot(2*num_wg,1,idx_electrode+1)       
    plt.plot(time_range_control, control_voltage[:,idx_electrode])
    plt.ylim([-6,7])
    plt.xlabel('t',fontsize=11)
    plt.ylabel("$V_%s(t)$"%idx_electrode, fontsize=11)
    plt.xticks([transition[0][0]for transition in sequence]+[sequence[-1][0][1]], fontsize=11)
    plt.yticks(np.arange(-5,6,2.5),fontsize=11)
    for transition in sequence:
        plt.arrow(transition[0][0],5,transition[0][1]-transition[0][0],0, length_includes_head=True,head_width=0.25, head_length=2.5e-3,color="red")
        plt.arrow(transition[0][1],5,transition[0][0]-transition[0][1],0, length_includes_head=True,head_width=0.25, head_length=2.5e-3,color="red")
        plt.text(0.5*(transition[0][0] + transition[0][1]), 5, transition[2], fontsize=11,horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white',edgecolor='red'))
    plt.grid()
plt.tight_layout()
plt.savefig('controller_complex_voltages.pdf', format='pdf', bbox_inches='tight')


# In[ ]:




