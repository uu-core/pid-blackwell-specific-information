import unittest
import numpy as np
import pandas as pd
import pid_implementation as pid

decimals = 6
measureoptions = ['KL', 'TV', 'C2', 'H2', 'LC', 'JS']
examples =  [('XOR',  	pd.DataFrame(np.array([[0,0,0,1/4],[0,1,1,1/4],[1,0,1,1/4],[1,1,0,1/4]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'), 	# Example XOR Figure 4 of Finn and Lizier (2018)
 			 ('PwUnq',  pd.DataFrame(np.array([[0,1,1,1/4],[1,0,1,1/4],[0,2,2,1/4],[2,0,2,1/4]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'), 	# Example PwUnq Figure 5 of Finn and Lizier (2018)
			 ('RdnErr',	pd.DataFrame(np.array([[0,0,0,3/8],[1,1,1,3/8],[0,1,0,1/8],[1,0,1,1/8]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'), 	# Example RdnErr Figure 6 of Finn and Lizier (2018)
			 ('Tbc',	pd.DataFrame(np.array([[0,0,0,1/4],[0,1,1,1/4],[1,0,2,1/4],[1,1,3,1/4]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'),		# Example Tbc Figure 7 of Finn and Lizier (2018)
			 ('Unq',	pd.DataFrame(np.array([[0,0,0,1/4],[0,1,0,1/4],[1,0,1,1/4],[1,1,1,1/4]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'),		# Example Unq Figure A2 of Finn and Lizier (2018)
			 ('AND',	pd.DataFrame(np.array([[0,0,0,1/4],[0,1,0,1/4],[1,0,0,1/4],[1,1,1,1/4]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'),		# Example AND Figure A3 of Finn and Lizier (2018)
			 ('Generic',pd.DataFrame(np.array([[0,0,0,0.0625],[0,0,1,0.3],[1,0,0,0.0375],[1,0,1,0.05],[0,1,0,0.1875],[0,1,1,0.15],[1,1,0,0.2125]]), columns=['V1', 'V2', 'T', 'Pr']), ['V1', 'V2'], 'T'), # Example of Section 3.1 of Mages and Rohner (2023) to highlight difference from Williams and Beer (2010)
			]

class TestCases(unittest.TestCase):
    ## compare redundancy and synergy decomposition results for all measures and examples
    def test_consistency(self):
        for measure in measureoptions:
            for name, pdDistribution, predictors, target  in examples:
                synRes = pid.SynergyLatticePID(predictors,target,pdDistribution,measure)
                synResults = {atom: atom.partial for atom in synRes.atoms}
                redRes = pid.RedundancyLatticePID(predictors,target,pdDistribution,measure)
                redResults = {atom: atom.partial for atom in redRes.atoms}
                self.assertAlmostEqual(synResults[pid.SynergyAtom([[]])], redResults[pid.RedundancyAtom([['V1'], ['V2']])], decimals, f'Redundancy missmatch in example {name} at measure {measure}')        # Redundancy
                self.assertAlmostEqual(synResults[pid.SynergyAtom([['V1']])], redResults[pid.RedundancyAtom([['V2']])], decimals, f'Unique V2 missmatch in example {name} at measure {measure}')             # Unique V2
                self.assertAlmostEqual(synResults[pid.SynergyAtom([['V2']])], redResults[pid.RedundancyAtom([['V1']])], decimals, f'Unique V1 missmatch in example {name} at measure {measure}')             # Unique V1
                self.assertAlmostEqual(synResults[pid.SynergyAtom([['V1'],['V2']])], redResults[pid.RedundancyAtom([['V1', 'V2']])], decimals, f'Synergy missmatch in example {name} at measure {measure}')  # Synergy

if __name__ == '__main__':
    unittest.main()