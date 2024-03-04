# Quantum Process Tomography of Structured Optical Gates with Convolutional Neural Networks
Code used for the generation of synthetic experiments for the Quantum Process Tomography (QPT) of complex SU(2) gates, as well as for the training of U-Nets on said examples. The results of this code are featured in our pre-print (https://arxiv.org/abs/2402.16616).  

# How to use
- datagen.py generates synthetic datasets of periodic/waveplate-based SU(2) processes.
- train.py creates and trains the neural network model. Can either use a pre-generated dataset or can generate the experiments in real-time. 
- datamash.py combines two datasets together. This is used, in particular, to create augmented datasets for two-step training 
- synthetic_utils.py holds all the functions related to the generation of synthetic processes and is used in datagen as well as in the data analysis notebooks (see below)  

Use of the datagen.py, train.py, and datamash.py scripts requires configuration of the 'datagen.yaml', 'train.yaml', and 'datamash_simple.yaml' configuration files, which can be found in the configs folder. The files can be configured to generate experiments and to train networks of arbitrary complexity.

# SLURM
This code was used principally in a Slurm-based environment provided by Compute Canada. We provide the scripts used to run the code therein in the startups folder. In the command line, executing sbatch <name of slurm script> runs the script, where: 

- startup_datagen runs DatagenLocal.py (this requests the use of CPU nodes)
- startup_train runs train.py (this requests the use of multiple GPU nodes)
- startup_datamash runs datamash.py (this requests the use of CPU nodes)

It is possible to run multiple instances of the above process in parallel by configuring or adding the --array line in each script, then modifying the .yaml names with an integer at the end (assume base 0 numbering).  

# Data Analysis
- The reconstruct_process.ipynb loads a neural network, loaded from the models folder, trained from our scheme to reconstruct selected processes synthetically (loaded from theoretical_data folder) and with real experimental data (found in the experimental_data folder). 
- The FigureOfMerit.ipynb loads a neural network to reconstruct many (on the order of $10^2$ - $10^3$) processes. It is also used to compare our result with the genetic algorithm described in (https://doi.org/10.1364/OE.491518) 
  

