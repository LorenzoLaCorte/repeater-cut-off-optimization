from itertools import chain, combinations
from repeater_types import checkAsymProtocol

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
    if f"s{segment-1}" not in final_sequence:
        return False
    swap_of_prev = final_sequence.index(f"s{segment-1}")

    # If the swap of the previous segment is after the current position, return False
    if swap_of_prev >= position:
        return False
    
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


def get_possible_combinations(swap_seq, dists):
    """
    Generate lists with the positions of the dists
    """
    all_dists = dists.copy()
    curr = all_dists.copy()

    all_possible_combs = [[]]
    for idx, step in enumerate(swap_seq):       
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
    not_complete_combs = [comb for comb in all_possible_combs if len(comb) < (len(swap_seq) + len(dists))]
    last_dist = max([int(dist[1:]) for dist in all_dists])

    for comb in not_complete_combs:
        for _ in range(len(swap_seq) + len(dists) - len(comb)):
            comb.append(f"d{last_dist}")
    
    # Return the unique combinations
    return [list(comb) for comb in set([tuple(comb) for comb in all_possible_combs])]


def generate_sequences(swap_seq, kappa):
    """
    Generate my own combs by considering spots where dists can be placed
    """
    if kappa == 0:
        yield swap_seq
        return
    
    S = len(swap_seq) + 1
    dists = [f"d{i}" for i in range(S) for _ in range(kappa)]

    possible_sequences = get_possible_combinations(swap_seq, dists)

    for sequence in possible_sequences:
        for idx, step in enumerate(sequence):
            if step.startswith("d"):
                if not checkUniqueness(int(step[1:]), idx, sequence): 
                    break
        else:
            # Check if the sequence is valid
            try:
                yield sequence
            except:
                pass

if __name__ == "__main__":
    swap_seq = [ "s1", "s0", "s2"]
    kappa = 2

    sequences = list(generate_sequences(swap_seq, kappa))
                    
    for seq in sequences:
        print(seq)
    print(len(sequences))
