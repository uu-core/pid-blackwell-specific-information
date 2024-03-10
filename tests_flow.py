import pid_implementation as pid
import chain_distributions as chain

check_margin = 1e-6
measureoptions = ['KL', 'TV', 'C2', 'H2', 'LC', 'JS']

print(f'-> Generating example distribution...')
#example_chain = chain.random_example()           # generate new random example
example_chain = chain.paper_example()             # use paper example


# check if flow from redundancy lattice and synergy lattice are equal
def compare_flow(red_flow,syn_flow,step):
    atomSynToRed  = lambda i:  {#f"[['X{i}', 'Y{i}']]": '[[]]',                             # bot       (<- these have different function on both lattices)
                                f"[['X{i}'], ['Y{i}']]": f"[['X{i}', 'Y{i}']]",             # synergy
                                f"[['X{i}']]": f"[['Y{i}']]",                               # unique Y
                                f"[['Y{i}']]": f"[['X{i}']]",                               # unique X
                                f'[[]]': f"[['X{i}'], ['Y{i}']]"}                           # redundancy
    # access by hash does no work well (inconsistent between instances), use names as dict-index instead
    red_res = {(str(r1),str(r2)): val for (r1,r2), val in red_flow.results.items()}
    syn_res = {(str(s1),str(s2)): val for (s1,s2), val in syn_flow.results.items()}
    for synStr1,redStr1 in atomSynToRed(step).items():
        for synStr2,redStr2 in atomSynToRed(step+1).items():
            if not (abs(red_res[(redStr1,redStr2)] - syn_res[(synStr1,synStr2)]) <= check_margin):
                print(f'mismatch at step {step}, redundancy atom {redStr1}->{redStr2} and synergy atom {synStr1}->{synStr2} ' +
                      f'with values {red_res[(redStr1,redStr2)]} and {syn_res[(synStr1,synStr2)]}')
                return False
    return True

# check if flow from redundancy lattice and synergy lattice are equal for the full chain
def compare_flow_results(redundancy_results, synergy_results):
    assert len(redundancy_results) == len(synergy_results)
    for i, (red_flow, syn_flow) in enumerate(zip(redundancy_results, synergy_results)):
        if not compare_flow(red_flow, syn_flow, i+1):
            return False
    return True

# compute flow
def compute_flows(latticePID,measure):
    results = []
    for step in range(1,example_chain.chain_length):
        print(f'-> start computing flow {step}...')
        results.append(pid.InformationFlow(latticePID, [f'X{step}', f'Y{step}'], [f'X{step+1}', f'Y{step+1}'], 'T', example_chain.joint_distribution, measure, prob_column_name='Pr',printPrecision=5,normalize_print=False))
        results[-1].compute()
    return results

# compute flows for all measures on both lattices and compare results
for i, measure in enumerate(measureoptions):
    print(f'\n --- analyzing measure {i+1}/{len(measureoptions)} ({measure}) ---\n')
    print(f'Computing flow from redundancy lattice of measure {measure}:')
    redundancy_results = compute_flows(pid.RedundancyLatticePID, measure)
    print(f'Computing flow from synergy lattice of measure {measure}:')
    synergy_results = compute_flows(pid.SynergyLatticePID, measure)

    assert compare_flow_results(redundancy_results, synergy_results)
    print('\n--> Both lattices obtained equivalent flows <--\n')

print('--------------------------------------\n-> All tests completed successfully <-\n--------------------------------------\n')