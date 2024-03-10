import numpy as np
import pandas as pd
import pid_implementation as pid
import chain_distributions as chain

# Computing all results from the publication: 
# "Non-Negative Decomposition of Multivariate Information: from Minimum to Blackwell Specific Information" by in Mages, Anastasiadi and Rohner (2024)
# 
# The results from using the redundancy or synergy lattice are identical (see test_flow.py & tests_pid.py -> used for checking the implementation)

# choose lattice (redundancy or synergy)
LatticePID = pid.RedundancyLatticePID
#LatticePID = pid.SynergyLatticePID  

measureoptions = ['KL', 'TV', 'C2', 'H2', 'LC', 'JS']

examples2 = [('XOR',  	pd.DataFrame(np.array([[0,0,0,1/4],[0,1,1,1/4],[1,0,1,1/4],[1,1,0,1/4]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'), 	# Example XOR Figure 4 of Finn and Lizier (2018)
 			 ('PwUnq',  pd.DataFrame(np.array([[0,1,1,1/4],[1,0,1,1/4],[0,2,2,1/4],[2,0,2,1/4]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'), 	# Example PwUnq Figure 5 of Finn and Lizier (2018)
			 ('RdnErr',	pd.DataFrame(np.array([[0,0,0,3/8],[1,1,1,3/8],[0,1,0,1/8],[1,0,1,1/8]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'), 	# Example RdnErr Figure 6 of Finn and Lizier (2018)
			 ('Tbc',	pd.DataFrame(np.array([[0,0,0,1/4],[0,1,1,1/4],[1,0,2,1/4],[1,1,3,1/4]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'),		# Example Tbc Figure 7 of Finn and Lizier (2018)
			 ('Unq',	pd.DataFrame(np.array([[0,0,0,1/4],[0,1,0,1/4],[1,0,1,1/4],[1,1,1,1/4]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'),		# Example Unq Figure A2 of Finn and Lizier (2018)
			 ('AND',	pd.DataFrame(np.array([[0,0,0,1/4],[0,1,0,1/4],[1,0,0,1/4],[1,1,1,1/4]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'),		# Example AND Figure A3 of Finn and Lizier (2018)
			 ('Generic',pd.DataFrame(np.array([[0,0,0,0.0625],[0,0,1,0.3],[1,0,0,0.0375],[1,0,1,0.05],[0,1,0,0.1875],[0,1,1,0.15],[1,1,0,0.2125]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'), # Example of Section 3.1 of Mages and Rohner (2023)
			]

examples3 = [('Tbep',	pd.DataFrame(np.array([[0,0,0,0,1/4],[0,1,1,1,1/4],[1,0,1,2,1/4],[1,1,0,3,1/4]]), columns=['V1', 'V2', 'V3', 'T', 'Pr']), ['V1', 'V2', 'V3'], 'T'), # Example IdentityCounterExample Theorem 2: Rauh et al. (2014) and Tbep Figure A1 of Finn and Lizier (2018)
			 ]

def figure_5_pid_comparison():
	for i,measure in enumerate(measureoptions):
		for name, pdDistribution, predictors, target  in examples2:
			print(f'\n ---- Decomposition example: {name} at measure: {measure} ---- ')
			res = LatticePID(predictors,target,pdDistribution, measure, normalize_print=True)
			print(res)

def figure_7_A3_flow_comparison():
	example_chain = chain.paper_example()       # use paper example

	# compute flow
	def compute_flows(latticePID, measure):
		results = []
		for step in range(1,example_chain.chain_length):
			print(f'-> start computing flow {step}...')
			results.append(pid.InformationFlow(latticePID, [f'X{step}', f'Y{step}'], [f'X{step+1}', f'Y{step+1}'], 'T', example_chain.joint_distribution, measure, prob_column_name='Pr',printPrecision=5,normalize_print=True,print_partial_zeros=False))
			results[-1].compute()
		return results

	# compute flows for all measures
	for i, measure in enumerate(measureoptions):
		print(f'\n --- flow analysis using measure {i+1}/{len(measureoptions)} ({measure}) ---\n')
		results = compute_flows(LatticePID, measure)		## Use redundancy lattice
		print('Results:')
		for i,x in enumerate(results):
			print(LatticePID([f'X{i+1}', f'Y{i+1}'], 'T', example_chain.joint_distribution, measure, printPrecision=5, normalize_print=True))
			print(f' -- flow results step: {i+1}->{i+2}')
			print(x)
		print(LatticePID([f'X{len(results)+1}', f'Y{len(results)+1}'], 'T', example_chain.joint_distribution, measure, printPrecision=5, normalize_print=True))


if __name__ == '__main__':
	print('\n_____ PID comparison (Figure 5) _____\n')
	figure_5_pid_comparison()
	print('\n_____ Flow comparison (Figure 7 & A3) _____\n')
	figure_7_A3_flow_comparison()
