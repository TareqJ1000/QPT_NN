# QPT_NN
Code used for the generation of synthetic experiments for the Quantum Process Tomography (QPT) of SU(2), as well as for the training of U-Nets on said examples. The results of this code are featured in our pre-print (https://arxiv.org/abs/2402.16616).  

# How to use
- DatagenLocal.py generates synthetic datasets of periodic/waveplate-based SU(2) processes.
- train.py creates and trains the neural network model. Can either use a pre-generated dataset, or can generate the experiments in real-time. 

Both codes requires configuration of, respectively, the 'datagen.yaml' and 'train.yaml' configuration files, which can be found in the configs folder. The files can be configured to generate experiments and to train networks of arbitrary complexity and to the users liking. 
  

