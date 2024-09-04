import itertools
from itertools import chain, combinations

from gp_utils import get_swap_dist_shape
from repeater_types import SimParameters, checkProtocolUnit, checkAsymProtocol

def checkUniqueness(segment, position, final_sequence):
    """
    For d{i} to have meaning in position p
    before it there must be either:
    - zero swaps
    - a decreasing sequence of swaps going from s{i-1} down
    """
    # if f"s{segment}" in final_sequence and final_sequence.index(f"s{segment}") < position:
    #     return False
    
    # If all elements before position are None or beginning with 'd', return True
    if all([final_sequence[j].startswith("d") for j in range(position)]):
        return True

    # Check if, from where s{i-1} happened, swapped segments start from i-1 and go down
    swap_of_prev = final_sequence.index(f"s{segment-1}")

    # Considering the sequence of swapped segments starting from s{i-1}
    swapped_segments = [int(final_sequence[i][1:]) for i in range(swap_of_prev, position) 
                        if final_sequence[i].startswith("s")]

    if segment-1 in swapped_segments and swapped_segments == sorted(swapped_segments, reverse=True):
        return True
    
    return False
        

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def get_possible_combinations(swap_seq, dists, total_slots):
    """
    Generate lists of length total_slots with the positions of the dists
    """
    all_dists = dists.copy()
    curr = all_dists.copy()

    all_possible_combs = [[]]
    for step in swap_seq:
        swapped_segment = int(step[1:])

        new_all_possible_combs = []
        for comb in all_possible_combs:
            # Consider all the possible subsequences at step before s{i}
            # - add all dists of the swapped segment here (after they are not valid anymore)
            curr = comb.copy() 
            
            remaining_dists = all_dists.copy()
            for el in curr:
                if el[0] == 'd':
                    remaining_dists.remove(el) # Remove ONLY ONE entry on el from remaining_dist
            curr += [dist for dist in remaining_dists if dist == f"d{swapped_segment}"]
            
            remaining_dists = [dist for dist in remaining_dists if dist != f"d{swapped_segment}"]
            
            # Build curr from prev
            # - update all_possible_combs by adding the one of the ones to be considered next
            new_combs = []
            for remaining_dist_comb in powerset(remaining_dists):
                new_combs.append(curr + list(remaining_dist_comb) + [f"s{swapped_segment}"])
            
            new_all_possible_combs += new_combs
        
        all_possible_combs = new_all_possible_combs

    # If the last dist has not been added, add it
    max_dist = max([int(dist[1:]) for dist in all_dists])
    for comb in all_possible_combs:
        if f"d{max_dist}" not in comb:
            comb.append(f"d{max_dist}")
    return all_possible_combs


def generate_sequences(swap_seq, dists):
    total_slots = len(swap_seq) + len(dists)
    dist_segments = [int(dist[1:]) for dist in dists]
    # TODO: instead of combs of range(total_slots), len(dists)
    # I have to generate my own combs by considering spots where dists can be placed
    possible_sequences = get_possible_combinations(swap_seq, dists, total_slots)

    for sequence in possible_sequences:
        for dist in dists:
            dist_segm = int(dist[1:])
            dist_idx = sequence.index(dist)
            if not checkUniqueness(dist_segm, dist_idx, sequence): 
                break
        else:
            # Check if the sequence is valid
            try:
                checkAsymProtocol(sequence)
                yield sequence
            except:
                pass

swap_seq = ["s0", "s2", "s1"]
dists = ["d0", "d1", "d2", "d3"]
# dists = ["d0", "d0", "d1", "d1", "d2", "d2", "d3", "d3"]

for seq in generate_sequences(swap_seq, dists):
    print(seq)

print(len(list(generate_sequences(swap_seq, dists))))
