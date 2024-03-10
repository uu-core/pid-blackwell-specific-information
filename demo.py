
import numpy as np
import pandas as pd
import pid_implementation as pid

# measure options: 'KL', 'TV', 'C2', 'H2', 'LC', 'JS'
measure = 'KL'

'''
   (1) PID Example
'''
distribution = pd.DataFrame(np.array([[0,0,0,1/4],[0,1,1,1/4],[1,0,1,1/4],[1,1,0,1/4]]), columns=['V1', 'V2', 'T', 'Pr'])  # specify joint distriution as dataframe
# Arguments: list of sources, target variable, joint distribution data-frame, measure-indicator, column name containing probability
redundancyTest = pid.RedundancyLatticePID(['V1','V2'], 'T', distribution, measure, prob_column_name='Pr')					   # compute results on redundancy lattice
print('\n......... Redundancy lattice example .........')
print(redundancyTest)		                                                                                               # print results

distribution = pd.DataFrame(np.array([[0,0,0,1/4],[0,1,1,1/4],[1,0,1,1/4],[1,1,0,1/4]]), columns=['V1', 'V2', 'T', 'Pr'])  # specify joint distriution as dataframe
redundancyTest = pid.SynergyLatticePID(['V1','V2'], 'T', distribution, measure, prob_column_name='Pr')					       # compute results on synergy lattice
print('......... Synergy lattice example .........')
print(redundancyTest)		                                                                                               # print results

'''
   (2) Information flow example
'''
distribution = pd.DataFrame(np.array([[0,0,0,0,0,1/8],
                                      [0,1,1,1,1,1/8],
                                      [0,0,0,2,2,1/8],
                                      [0,1,1,3,3,1/8],
                                      [1,0,2,0,2,1/8],
                                      [1,1,3,1,3,1/8],
                                      [1,0,2,2,0,1/8],
                                      [1,1,3,3,1,1/8]]), columns=['A', 'B', 'X', 'Y', 'C', 'Pr'])

# FLow example based on the redundancy lattice: sources lattice 1, sources lattice 2, target, distribution, measure-indicator, probability-column name, no printing of zero flows
flow = pid.InformationFlow(pid.RedundancyLatticePID, ['A', 'B','Y'], ['X','Y'], 'C',distribution,measure,prob_column_name='Pr', print_partial_zeros=False)
print('......... Redundancy lattice flow example .........')
print(flow)
print('Entropy (C):\t', flow.entropy(['C']))
print('Entropy (X,Y):\t', flow.entropy(['X','Y']),end='\n\n')

# FLow example based on the synergy lattice:
flow = pid.InformationFlow(pid.SynergyLatticePID, ['A', 'B','Y'], ['X','Y'], 'C',distribution,measure,prob_column_name='Pr', print_partial_zeros=False)
print('......... Synergy lattice flow example .........')
print(flow)
print('Entropy (C):\t', flow.entropy(['C']))
print('Entropy (X,Y):\t', flow.entropy(['X','Y']))