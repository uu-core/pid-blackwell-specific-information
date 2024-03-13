'''
An implementation of the Partial Information Decomposition (PID) of:
	"Non-Negative Decomposition of Multivariate Information: from Minimum to Blackwell Specific Information" by T. Mages, E. Anastasiadi and C. Rohner (2024)

References:
	Finn, C., 2019. A New Framework for Decomposing Multivariate Information. Ph.D. thesis. University of Sydney.
	Chicharro, D.; Panzeri, S. Synergy and Redundancy in Dual Decompositions of Mutual Information Gain and Information Loss. Entropy 2017
	Williams, P.L., Beer, R.D., 2010. Nonnegative decomposition of multivariate information. arXiv:1004.2515.

Comments:
	- both (redundancy and synergy) are obtained from their pointwise measures - computations on the synergy lattice are more effcient.

Example usage:
	import numpy as np
	import pandas as pd
	import pid_implementation as pid

	distribution = pd.DataFrame(np.array([[0,0,0,1/4],[0,1,1,1/4],[1,0,1,1/4],[1,1,0,1/4]]), columns=['V1', 'V2', 'T', 'Pr'])  # specify joint distriution as dataframe
	redundancyTest = pid.RedundancyLatticePID(['V1','V2'], 'T', distribution, 'KL', prob_column_name='Pr')					   # compute results
	print(redundancyTest)																									   # print results
'''

from itertools import combinations, product
from functools import reduce, cache
from math import sqrt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

logBase = 2
precision = 16
log = lambda x: np.log(x)/np.log(logBase) if abs(x) >= 10**(-precision) else 0  # for convention 0*log(0) = 0 in 'KL' and 'JS' measures

# List of f-divergence functions (available measure options):
HD = lambda x,a: (x**a-1)/(a-1)										# Hellinger divergence (give 'a' constant value, the function shall only take one argument. See example below)
f_options = {'KL': lambda x: x*log(x) ,								# Kullback–Leibler divergence
			 'TV': lambda x: 0.5*abs(x-1) ,							# Total variation distance
			 'C2': lambda x: (x-1)**2,								# (Pearson) \chi^2 divergence
			 'H2': lambda x: (1-sqrt(x))**2,						# Squared Hellinger distance
			 'LC': lambda x: (1-x)/(2*x+2),							# Le Cam distance
			 'JS': lambda x: x*log((2*x)/(x+1)) + log(2/(x+1)),		# Jensen-Shannon divergence
			 #'HD-0.5': lambda x: HD(x,0.5),						# Example for adding a new measure: Hellinger divergence with 'a=0.5' is available under the name 'HD-0.5' when uncommenting this line.
}


###############################################################
##### 1. Basic Definitions
###############################################################

''' predictors: list of predictor variables e.g. ['V1', 'V2', 'C3'] '''
def P1(predictors): # Sources: power set without the empty set
    return [list(source) for r in range(1,len(predictors)+1) for source in combinations(predictors, r)]

#--> subseteq for axiom 2: Monotonicity             <--#
''' A, B: sources (list of predictors) e.g. A=[['V1', 'V2'],['C3']] '''
def subseteq(A,B): # True if A subseteq of B 
	return all([a in B for a in A])

#--> PI-atom queality:                            <--#
''' alpha, beta: collections of sources e.g. alpha=[['V1', 'V2'],['C3']] '''
def equalAtom(alpha, beta):
	return all([any([set(a)==set(b) for b in beta]) for a in alpha] + [any([set(a)==set(b) for a in alpha]) for b in beta])

#--> Equation 4      of Williams and Beer (2010)  <--#
#--> Equation 2.17   of Thesis Finn (2019)        <--#
# modified to include emptyset (to be used as both gain/loss lattice)
''' predictors: list of predictor variables e.g. ['V1', 'V2', 'C3'] '''
def A(predictors): # all relevant collection of sources from the predictors
	return [x for x in P1(P1(predictors)) if not any([subseteq(A1,A2) for (A1, A2) in combinations(x,2)])]
	
#--> Equation 5      of Williams and Beer (2010)  <--#
#--> Equation 2.18   of Thesis Finn (2019)        <--#
''' alpha,beta: collections of sources e.g. alpha=[['V1', 'V2'],['C3']] '''
def red_leq(alpha,beta): # less or equal for two collections of sources
	return all([any([subseteq(A,B) for A in alpha]) for B in beta])

#--> Equation 13   of Chicharro and Panzeri (2017) <--#
''' alpha,beta: collections of sources e.g. alpha=[['V1', 'V2'],['C3']] '''
def syn_leq(alpha,beta): # less or equal for two collections of sources
	return all([any([subseteq(B,A) for A in alpha]) for B in beta])


###############################################################
##### 2. Atom Definition (redundancy / syngergy)
###############################################################
class Atom:
	def __init__(self, atom, leq_fun):
		self.list =  atom
		self.cumulative = None
		self.partial = None
		self.leq_fun = leq_fun
	def __le__(self,other):
		return self.leq_fun(self.list, other.list)
	def __ge__(self,other):
		return (other.__le__(self))
	def __eq__(self,other):
		return (self <= other) and (other <= self)
	def __ne__(self,other):
		return not (self == other)
	def __lt__(self,other):
		return (self <= other) and not (self == other)
	def __gt__(self,other):
		return (other <= self) and not (self == other)
	def __or__(self,other): # check if incomparable 			(atom1 | atom2)
		return (not (self <= other)) and (not (other <= self))
	def __xor__(self,other): # meet of two atoms	 			(atom1 ^ atom2)
		full_atom = Atom(self.list + other.list,self.leq_fun)
		reduced_list = self.list + other.list
		x = 0
		while x < len(reduced_list): # remove one source after the other, as long as it maintaince equivalence
			if Atom(reduced_list[:x] + reduced_list[x+1:],self.leq_fun) == full_atom:
				del reduced_list[x]	 # shorten list
			else:
				x +=1				 # increase index
		return Atom(reduced_list,self.leq_fun)
	def __repr__(self):
		return f'{self.list}'
	def __hash__(self):
		return frozenset({frozenset(x) for x in self.list}).__hash__()
	
class RedundancyAtom(Atom):
	def __init__(self, atom):
		super().__init__(atom,red_leq)

	@staticmethod
	def bot(_):
		return Atom([[]],red_leq)  # bottom element

class SynergyAtom(Atom):
	def __init__(self, atom):
		super().__init__(atom,syn_leq)

	@staticmethod
	def bot(preds):
		return Atom([preds],syn_leq)  # bottom element for lattice of preds

###############################################################
##### 3. Lattice Definition (redundancy / syngergy)
#####    - based on pandas dataframe representing joint distribution
###############################################################
class Lattice:
	def __init__(self, predictor_variables, target_variable, atom_class, distribution, pw_measure, prob_column_name, printPrecision, normalize_print, print_bottom, print_partial_zeros, compute_automatically):
		assert set(predictor_variables+[target_variable, prob_column_name]).issubset(set(distribution.columns)) # the provided variables must be defined in the given distribution
		self.printPrecision = printPrecision
		self.predictor_variables = predictor_variables
		self.target_variable = target_variable
		vars = list(set(predictor_variables + [target_variable]))
		self.distribution = distribution[vars + [prob_column_name]].copy(deep=False).reset_index(drop=True)
		self.distribution = self.distribution.groupby(vars, as_index=False)[prob_column_name].sum() # remove unused variables
		self.pw_measure = pw_measure
		self.A_func = lambda preds: A(preds) +[[[]]]
		self.prob_column_name = prob_column_name
		self.atom_class = atom_class
		self.normalize_print = normalize_print # if true, normalize to entropy of target variable
		self.print_bottom = print_bottom
		self.print_partial_zeros = print_partial_zeros
		if compute_automatically:
			self.construct_atoms()
			self.compute_inverse()
	#
	def __repr__(self):
		normalization = self.entropy([self.target_variable]) if self.normalize_print else 1.0
		maxLen = max([len(str(x)) for x in self.atoms])
		res = ''
		if self.atoms[0].partial == None:
			res += '-- The decomposition has not been computed yet (call: self.compute_inverse) --\n'
		atoms = sorted(self.atoms,reverse=True)
		res += f'{"atom".center(maxLen,"-")}-:-{"cumulative".center(12,"-")}-|-{"partial".center(12,"-")} |\n'
		for x in atoms:
			if (self.print_bottom or x != self.atom_class.bot(self.predictor_variables)) and (self.print_partial_zeros or x.partial > 0):
				res += f'{str(x).center(maxLen," ")} : {0.0 + round(x.cumulative/normalization,self.printPrecision):>12} | {0.0 + round(x.partial/normalization,self.printPrecision):>12} |\n'
		return res
	#
	def entropy(self,variable_list):
		dist = self.distribution.agg(lambda row: pd.Series({'X':'-'.join([str(row[x]) for x in variable_list]), 'Pr': row[self.prob_column_name]}), axis='columns').copy(deep=True)	# construct joint distribution
		dist = dist.groupby(['X'],as_index=False).sum() # combine identical states
		channel = np.array([[1,0],[0,1]])
		return sum([Px*self.pw_measure(Px, [channel]) for Px in dist['Pr'].tolist()]) # compute expected value of measuring the identity channel
	#
	def strictDownset(self,atom):
		assert atom in self.atoms
		return [x for x in self.atoms if (x < atom)]
	#
	def strictUpset(self,atom):
		assert atom in self.atoms
		return [x for x in self.atoms if (atom < x)]
	#
	def construct_atoms(self):
		self.atoms = [self.atom_class(x) for x in self.A_func(self.predictor_variables)]
	#
	def compute_inverse(self):
		# compute cumulative terms
		for atom in self.atoms:
			atom.cumulative = self.cumulative_measure(atom)
		# compute Möbius inverse
		for atom in self.atoms:
			atom.partial = self.möbius_inverse(atom)
	#
	@cache
	def möbius_inverse(self,atom):
		if atom.partial == None:
			atom.partial = atom.cumulative - sum([self.möbius_inverse(down) for down in self.strictDownset(atom)])
		return atom.partial
	#
	def cumulative_measure(self,atom): # expected value of pointwise lattices
		Pt = lambda state: self.distribution.groupby([self.target_variable]).sum()[self.prob_column_name][state]
		# compute expected value of valuation results (combining pointwise lattices)
		if self.atom_class == RedundancyAtom:
			# inclusion-exclusion relation contained in provided redundancy measure
			return sum([Pt(t)*self.pw_measure(Pt(t), [self.get_BinaryChannel(preds,t) for preds in atom.list]) for t in set(self.distribution[self.target_variable])])
		else:
			# covert to loss function quantifying bottom minus atom
			return sum([Pt(t)*(self.pw_measure(Pt(t),[self.get_BinaryChannel(self.predictor_variables,t)]) - self.pw_measure(Pt(t), [self.get_BinaryChannel(preds,t) for preds in atom.list])) for t in set(self.distribution[self.target_variable])])
	#
	def get_BinaryChannel(self,preds,t): 
		# construct joint distribution of preds with target variable
		pdPointwise = self.distribution.agg(lambda row: pd.Series({'V':'-'.join([str(row[x]) for x in preds]),self.target_variable: row[self.target_variable] == t, 'Pr': row[self.prob_column_name]}), axis='columns').copy(deep=True)
		# combine identical states
		pdPointwise = pdPointwise.groupby(['V',self.target_variable],as_index=False).sum()
		# construct numpy conditional distribution from pandas joint distribution
		p = pdPointwise[pdPointwise[self.target_variable] == True]['Pr'].sum()                # compute p = P(T==t)
		pdPointwise.loc[pdPointwise[self.target_variable] == True, 'Pr'] /= p                 # get conditional probability for T==t: x/p
		pdPointwise.loc[pdPointwise[self.target_variable] == False, 'Pr'] /= (1-p)            # get conditional probability for T!=t: x/(1-p)
		return pdPointwise.pivot_table(index=[self.target_variable],columns='V',values='Pr',fill_value=0).sort_index(ascending=False).to_numpy()

class SynergyLattice(Lattice):
	def __init__(self, predictor_variables,target_variable,distribution,cumulative_measure,prob_column_name='Pr',printPrecision=5,normalize_print=False,print_bottom=False,print_partial_zeros=True,compute_automatically=True):
		self.lattice_type_indicator = f'--> Synergy Lattice Decomposition (normalized to target entropy: {normalize_print}) <--'
		super().__init__(predictor_variables,target_variable,SynergyAtom,distribution,cumulative_measure,prob_column_name,printPrecision,normalize_print,print_bottom,print_partial_zeros,compute_automatically)

	def __repr__(self):
		return self.lattice_type_indicator + '\n' + super().__repr__()

class RedundancyLattice(Lattice):
	def __init__(self, predictor_variables,target_variable,distribution,cumulative_measure,prob_column_name='Pr',printPrecision=5,normalize_print=False,print_bottom=False,print_partial_zeros=True,compute_automatically=True):
		self.lattice_type_indicator = f'--> Redundancy Lattice Decomposition (normalized to target entropy: {normalize_print}) <--'
		super().__init__(predictor_variables,target_variable,RedundancyAtom,distribution,cumulative_measure,prob_column_name,printPrecision,normalize_print,print_bottom,print_partial_zeros,compute_automatically)

	def __repr__(self):
		return self.lattice_type_indicator + '\n' + super().__repr__()


###############################################################
##### 4. Measure construction
#####    - based on numpy array representing pointwise 
#####      conditional distribuation
###############################################################

def set_precision_variable(prec): # change precision of pointwise channel conversions
	global precision
	precision = prec

# it returns a function that takes a list of binary channels and measure their convex hull
def pw_measure_constructor(option):
	assert option in f_options, f"The provided option '{option}' is not available.\nPlease select one of the implemented measures: {f_options.keys()}"
	f = lambda x: f_options[option](x)
	r = lambda p_, x_, y_: 0 if x_ == 0 and y_ == 0 else (p_*x_ + (1 - p_)*y_)*f(x_/(p_*x_ + (1 - p_)*y_))
	i = lambda p_, channel: sum([r(p_,x,y) for (x,y) in np.round(channel.T,precision)])
	return i

'''
	returns the Blackwell-joint	for all pointwise channels in the list
'''
def BlackwellJoint_pw(pw_channel_list):
	def channelToPoints(k):
		points = sorted(k.T.tolist(),key=lambda x: x[1]/x[0] if x[0] != 0 else 10**15, reverse=True) # sort entries by likelihood ratio
		curve = [np.array([0,0])]
		for p in points:																			 # accumulate components to get the curve
			curve.append((np.array(p) + curve[-1]))
		return np.array(curve)
	#
	def hullToChannel(hull_points):
		cornerPoints = sorted(hull_points,key=lambda x: tuple(x))
		channel, init = [], np.array([0,0])
		for x in cornerPoints:
			if np.any(x != init):
				channel.append(np.array(x)-init)
				init = np.array(x)
		return np.array(channel).T.clip(0)
	#
	def myConvexHull(points):
		if all([abs(a-b) < 10**(-precision) for a,b in points.tolist()]):
			return np.array([[0,0],[1,1]])
		else:
			return np.unique(np.vstack([points[s] for s in ConvexHull(points).simplices]),axis=0)
	#
	def BlackwellJoint2(k1,k2):
		prec = precision
		k1,k2=np.round(k1,prec),np.round(k2,prec)
		# let's check for rounding issues from the conversion to pointwise channels. There shouldn't be any though...
		while not (np.all(k1 >= 0.0) and np.all(k2 >= 0.0)) and precision-prec<=2:
			prec-=1
			k1,k2=np.round(k1,prec),np.round(k2,prec)
			print(f' -> Warning, one pointwise channel is computed with lower precision ({prec}) due to a rounding issue.')
		assert np.all(k1 >= 0.0) and np.all(k2 >= 0.0), f'invalid channel {k1} or {k2}'
		return hullToChannel(myConvexHull(np.vstack((channelToPoints(k1),channelToPoints(k2)))).tolist())
	#
	bottom_element = np.array([[1],[1]])
	return reduce(BlackwellJoint2, pw_channel_list, bottom_element)

def pw_redundandy_measure(measure_option):
	g = pw_measure_constructor(option=measure_option)
	return lambda p, pw_channel_list: sum([((-1)**(len(sub_list)-1) * g(p,BlackwellJoint_pw(sub_list))) for sub_list in P1(pw_channel_list)])  # inclusion-exclusion principle (dual decomposition of synergy lattice)

def pw_synergy_measure(measure_option):
	g = pw_measure_constructor(option=measure_option)
	return lambda p, pw_channel_list: g(p,BlackwellJoint_pw(pw_channel_list))


###############################################################
##### 5. Partial Information Decomposition
###############################################################

class RedundancyLatticePID(RedundancyLattice):
	def __init__(self, predictor_variables,target_variable,distribution,measure_option,prob_column_name='Pr',printPrecision=5,normalize_print=False,print_bottom=False,print_partial_zeros=True,compute_automatically=True):
		super().__init__(predictor_variables,target_variable,distribution,pw_redundandy_measure(measure_option),prob_column_name,printPrecision,normalize_print,print_bottom,print_partial_zeros,compute_automatically)

class SynergyLatticePID(SynergyLattice):
	def __init__(self, predictor_variables,target_variable,distribution,measure_option,prob_column_name='Pr',printPrecision=5,normalize_print=False,print_bottom=False,print_partial_zeros=True,compute_automatically=True):
		super().__init__(predictor_variables,target_variable,distribution,pw_synergy_measure(measure_option),prob_column_name,printPrecision,normalize_print,print_bottom,print_partial_zeros,compute_automatically)


####################################################################################
##### 6. Information Flow Analysis (one step)
####################################################################################

# assumes Markov chain: target_variable -> predictor_list1 -> predictor_list2
class InformationFlow():
	def __init__(self, LatticePID, predictor_list1, predictor_list2, target_variable,distribution,measure_option,prob_column_name='Pr',printPrecision=5,normalize_print=False,print_partial_zeros=True,print_bottom=False):
		self.predictor_list1 = predictor_list1
		self.predictor_list2 = predictor_list2
		self.target_variable = target_variable
		vars = list(set(predictor_list1 + predictor_list2 + [target_variable]))
		self.distribution = distribution[vars + [prob_column_name]].copy(deep=False).reset_index(drop=True)
		self.distribution = self.distribution.groupby(vars, as_index=False)[prob_column_name].sum() # remove unused variables
		self.measure_option = measure_option
		self.prob_column_name = prob_column_name
		self.printPrecision = printPrecision
		self.normalize_print = normalize_print
		self.LatticePID = LatticePID
		self.lattice1 = LatticePID(predictor_list1,target_variable,distribution,measure_option,prob_column_name,printPrecision,normalize_print,print_bottom,print_partial_zeros,compute_automatically=False)
		self.lattice2 = LatticePID(predictor_list2,target_variable,distribution,measure_option,prob_column_name,printPrecision,normalize_print,print_bottom,print_partial_zeros,compute_automatically=False)
		self.lattice1.construct_atoms()
		self.lattice2.construct_atoms()
		self.lattice_combined = LatticePID(predictor_list1+predictor_list2,target_variable,distribution,measure_option,prob_column_name,printPrecision,normalize_print,print_bottom,print_partial_zeros,compute_automatically=False)
		self.bot_atom = lambda x: self.lattice_combined.atom_class.bot(x)
		self.print_bottom = print_bottom
		self.print_partial_zeros = print_partial_zeros
		self.results = None

	@cache
	def measure_cc(self,atom):
		return self.lattice_combined.cumulative_measure(atom)
		
	@cache
	def measure_pc(self,atom1,atom2):
		down1 = self.lattice1.strictDownset(atom1)
		return self.measure_cc(atom1 ^ atom2) - sum([self.measure_pc(x,atom2) for x in down1])

	@cache
	def measure_pp(self,atom1,atom2):
		down2 = self.lattice2.strictDownset(atom2)
		return self.measure_pc(atom1,atom2) - sum([self.measure_pp(atom1,x) for x in down2])
	
	def compute(self):
		self.results = {}
		for (atom1,atom2) in product(self.lattice1.atoms,self.lattice2.atoms):
			self.results[(atom1,atom2)] = self.measure_pp(atom1,atom2)

	def __repr__(self):
		if not self.results:
			self.compute()
		bot1, bot2 = self.bot_atom(self.predictor_list1), self.bot_atom(self.predictor_list2)
		scale = self.entropy([self.target_variable]) if self.normalize_print else 1.0
		res = f' {self.lattice1.lattice_type_indicator}\n Flow from {self.predictor_list1} -> {self.predictor_list2} with respect to {self.target_variable} (normalized to target entropy: {self.normalize_print})\n'
		res += f' Target variable entropy: {self.entropy([self.target_variable]):.4f}\n'
		maxLen1, maxLen2 = max([len(str(x[0])) for x in self.results]), max([len(str(x[1])) for x in self.results])
		for (atom1,atom2), value in self.results.items():
			if (self.print_bottom or (atom1 != bot1 and atom2 != bot2)) and (self.print_partial_zeros or value > 1e-8):
				res += f'{str(atom1).center(maxLen1+2," ")} -> {str(atom2).center(maxLen2+2," ")} : {0.0 + round(value/scale,self.printPrecision)}\n'
		return res
	
	def entropy(self,variable_list):
		self.full_lattice = self.LatticePID(self.predictor_list1+self.predictor_list2,self.target_variable,self.distribution,self.measure_option,self.prob_column_name,self.printPrecision,self.normalize_print,compute_automatically=False)
		return self.full_lattice.entropy(variable_list)
