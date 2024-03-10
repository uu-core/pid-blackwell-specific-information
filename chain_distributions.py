import numpy as np
import pandas as pd

'''
Generate a random Markov Chain (X1,Y1) -> (X2,Y2) -> ... -> (Xn,Yn).
For simplicity, the random variables always have three/two possible events (|Xi| = 3 and |Yi| = 2).
'''

# generate a new random example for the flow analysis (the paper example can be found below)
class random_example:
	def __init__(self, min_row_value = 0.8, chain_length = 5, target_variables = ['X3', 'Y3'], margin = 1e-6, compute=True):
		self.min_row_value = min_row_value
		self.chain_length = chain_length
		self.target_variables = target_variables
		self.margin = margin
		self.index_dict = {# (X=0|1|2, Y=0|1): idx,
							(0,0):0, (0,1):1,
							(1,0):2, (1,1):3,
							(2,0):4, (2,1):5}
		self.input_distribution = None
		self.transition_matrix = None
		self.joint_distribution = None
		if compute:
			while self.joint_distribution is None: # just in case there is some rounding issue
				self.input_distribution = self.generate_transision_matrix(row_count=1)[0] 
				self.transition_matrix  = self.generate_transision_matrix(row_count=len(self.index_dict))
				self.joint_distribution = self.generate_joint_distribution(self.input_distribution, self.transition_matrix, self.chain_length)

	def generate_transision_matrix(self,row_count):
		res = []
		while len(res) < row_count:
			x = np.random.standard_gamma(1,size=len(self.index_dict))
			x = np.round(x/sum(x),2)
			if sum(x) == 1 and max(x) > self.min_row_value:
				res.append(x)
		return np.array(res)
	
	def generate_joint_distribution(self, input_distribution, transition_matrix, chain_length):		
		def construct_chain(last_joint_distribution, step):
			if step <= chain_length:
				# construct new dataframe
				last_cols = [x for x in last_joint_distribution.columns if x != 'Pr']
				cols = last_cols + [f'X{step}', f'Y{step}'] + ['Pr']
				new_rows = [list(row) + list(key) + 																															# state
							[last_joint_distribution.loc[idx,'Pr'] * transition_matrix[self.index_dict[tuple(row[[f'X{step-1}', f'Y{step-1}']])],self.index_dict[key]]] 	    # compute probability
							for idx, row in last_joint_distribution[last_cols].iterrows() for key in self.index_dict]															# loop
				new_joint_distribution = pd.DataFrame(new_rows,columns=cols)
				return construct_chain(new_joint_distribution, step+1)
			else:
				return last_joint_distribution
			
		# compute distribution
		init_joint_distribution = pd.DataFrame([list(key) + [input_distribution[self.index_dict[key]]] for key in self.index_dict],columns=['X1', 'Y1', 'Pr'])
		result = construct_chain(init_joint_distribution, 2)
		# check for rounding
		if abs(1 - sum(result['Pr'])) <= self.margin:
			joint_distribution = result[result['Pr']!= 0].copy(deep=False).reset_index(drop=True)
			# construct joint target variable
			joint_distribution['T'] = joint_distribution[self.target_variables].astype(str).agg('-'.join, axis=1)
			return joint_distribution
		else:
			return None


class paper_example(random_example):
	def __init__(self):
		super().__init__(min_row_value = 0.8, chain_length = 5, target_variables = ['X3', 'Y3'], margin = 1e-6, compute = False)
		self.input_distribution = np.array([0.01, 0.81, 0.  , 0.02, 0.09, 0.07])
		self.transition_matrix  = np.array([[0.05, 0.01, 0.04, 0.82, 0.02, 0.06],
											[0.05, 0.82, 0.  , 0.01, 0.06, 0.06],
											[0.04, 0.01, 0.82, 0.05, 0.04, 0.04],
											[0.03, 0.84, 0.02, 0.06, 0.04, 0.01],
											[0.04, 0.03, 0.03, 0.02, 0.06, 0.82],
											[0.07, 0.04, 0.01, 0.03, 0.81, 0.04]])
		self.joint_distribution = self.generate_joint_distribution(self.input_distribution, self.transition_matrix, 5, )