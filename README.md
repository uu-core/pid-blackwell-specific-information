# Implementation: Partial Information Decomposition - Blackwell Specific Information

This repository provides an implementation of the method and examples in:

_"Non-Negative Decomposition of Multivariate Information: from Minimum to Blackwell Specific Information"_ by T. Mages, E. Anastasiadi and C. Rohner (2024)

_When using this implementation, please cite the [corresponding publication](https://www.preprints.org/manuscript/202403.0285)_.

## Overview
- `demo.py`: provides a simple usage example
- `run_paper_examples.py`: computes and prints all results/examples found in the publication
- `pid_implementation.py`: provides the implementation of the presented method and flow analysis
- `chain_distributions.py`: provides a random markov chain and the used markov chain of the publication
- `tests_pid.py`: compares the results between the redundancy and synergy lattice for PID examples (used for implementation testing)
- `tests_pid.py`: compares the results between the redundancy and synergy lattice in information flow analyses (random or paper example; used for implementation testing)

## Interface
`RedundancyLatticePID`/`SynergyLatticePID` take the followign arguments:
- `predictor_variables` (list of predictor variables / columns in `distribution`)
- `target_variable` (name of the target variable / column in `distribution`)
- `distribution` (pandas dataframe containing the joint distribution)
- `measure_option` (see table below)
- `prob_column_name` (default=`'Pr'`, dataframe column name containing the probability)
- `printPrecision` (default=`5`, number of decimal positions in the printed output)
- `normalize_print` (default=`False`, normalize the result to the entropy of the target variable)
- `print_bottom` (default=`False`, print the bottom element of the lattice in results)
- `print_partial_zeros` (default=`True`, print all results, otherwise exclude those with no partial contribution)
- `compute_automatically` (default=`True`, internal setup routines - recommended to leave at default)

`InformationFlow` takes the same arguments as above, with the following differences:
- `LatticePID` (provide decomposition lattice: `RedundancyLatticePID` or `SynergyLatticePID`)
- `predictor_list1` (list of predictor variable at stage i)
- `predictor_list2` (list of predictor variable at stage i+1)

Measure options correspond to the used $f$-divergence:
| Measure option | Description |
|---|---|
| `'KL'` | Kullback–Leibler divergence (Mutual Information) |
| `'TV'` | Total variation distance |
| `'C2'` | $\chi^2$-divergence, Pearson |
| `'H2'` | Squared Hellinger distance |
| `'LC'` | Le Cam distance |
| `'JS'` | Jensen-Shannon divergence |

Additional measures can be added using the dictionary `f_options` in `pid_implementation.py`.

### Note:
- The decompositions of the redundancy and synergy lattice are both implemented based on their pointwise measures. The decomposition of the synergy lattice is more efficient.
- To compute Rényi-decompositions, add the Hellinger-divergence in `pid_implementation.py` with a constant parameter to the measure options (indicated by comment/example in `f_options`). Then you can use it to compute results and transfom them to Rényi-information as described in the publication.

## Example usage
```
import numpy as np
import pandas as pd
import pid_implementation as pid

# define the joint distribution        V1, V2,  T,  Pr
distribution = pd.DataFrame(np.array([[ 0,  0,  0, 1/4],
                                      [ 0,  1,  1, 1/4],
                                      [ 1,  0,  1, 1/4],
                                      [ 1,  1,  0, 1/4]]), columns=['V1', 'V2', 'T', 'Pr'])

'''
Construct information decomposition
'''
# define the decomposition (redundancy lattice)
redTest = pid.RedundancyLatticePID(['V1','V2'],'T',distribution, 'KL')
print(redTest)                                              # view results
print("Entropy (V1,V2):\t", redTest.entropy(['V1','V2']))   # compute the entropy of some variables

# define the decomposition (synergy lattice)
synTest = pid.SynergyLatticePID(['V1','V2'],'T',distribution, 'KL')
print(synTest)                                              # view results

'''
Construct information flow analysis of a Markov chain
'''
# define joint distribution of variables
distribution = pd.DataFrame(np.array([[0,0,0,0,0,1/8],
                                      [0,1,1,1,1,1/8],
                                      [0,0,0,2,2,1/8],
                                      [0,1,1,3,3,1/8],
                                      [1,0,2,0,2,1/8],
                                      [1,1,3,1,3,1/8],
                                      [1,0,2,2,0,1/8],
                                      [1,1,3,3,1,1/8]]), columns=['A', 'B', 'X', 'Y', 'C', 'Pr'])

flow = pid.InformationFlow(pid.RedundancyLatticePID, ['A', 'B','Y'], ['X','Y'], 'C', distribution,
                           'KL', prob_column_name='Pr', print_partial_zeros=False)
                            # don't print zero flows for shorter output list
print(flow)
print('Entropy (C):\t', flow.entropy(['C']))
```
