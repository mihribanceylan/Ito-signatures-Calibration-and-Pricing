import ast
import numpy as np
import esig

def get_keys_and_tuples(d_aug, level):
    """
    Converts the signature keys into a list of tuples.
    
    Parameters:
    - d_aug: The augmented dimension of the process (W, C).
    - level: The depth of the signature.
    
    Returns:
    - A list of tuples representing the keys.
    """
    keys_str = esig.sigkeys(d_aug, level).split()
    keys_tuple = []
    for k in keys_str:
        val = ast.literal_eval(k)
        if isinstance(val, int):
            val = (val,)
        elif not isinstance(val, tuple):
            raise TypeError(f"Unexpected key: {k}")
        keys_tuple.append(val)
    return keys_tuple


def generate_ito_correction_map(d_aug, d_loc, level):
    """
    Generates the Itô correction map for the given depth and dimensions.
    
    Parameters:
    - d_aug: The augmented dimension of the process.
    - d_loc: The dimension of the local process.
    - level: The depth of the signature.
    
    Returns:
    - keys: The list of keys for the signature.
    - cmap: A dictionary containing the correction map.
    """
    keys = get_keys_and_tuples(d_aug, level)
    qv_pairs = [(a,b) for a in range(2, d_loc+2) for b in range(a, d_loc+2)]

    def map_to_qv_channel(i, j):
        pair = (min(i,j), max(i,j))
        return (2 + d_loc) + qv_pairs.index(pair)

    def non_adjacent_subsets(idxs):
        res = [[]]
        for idx in idxs:
            res += [s+[idx] for s in res if (not s) or (idx - s[-1] > 1)]
        return res[1:]

    cmap = {}
    for key in keys:
        if key == ():
            cmap[key] = []
            continue
        idxs = tuple(int(x) for x in key)
        cand = [j for j in range(len(idxs)-1)
                if 2 <= idxs[j] <= d_loc+1 and 2 <= idxs[j+1] <= d_loc+1]
        combinations = []
        for subset in non_adjacent_subsets(cand):
            new_index = []
            skip = False
            for j in range(len(idxs)):
                if skip:
                    skip = False
                    continue
                if j in subset:
                    new_index.append(map_to_qv_channel(idxs[j], idxs[j+1]))
                    skip = True
                else:
                    new_index.append(idxs[j])
            combinations.append((tuple(new_index), (-0.5)**len(subset)))
        cmap[key] = combinations
    return keys, cmap


def ito_from_stratonovich(S_augmented, d_loc, level):
    """
    Converts Stratonovich signatures to Itô signatures.

    Parameters:
    - S_augmented: Augmented stream with time, values, and quadratic variations
    - d_loc: The number of assets
    - level: The depth of the signature

    Returns:
    - ito: The corrected Itô signature
    - strat_sig: The  Stratonovich signature
    """
    S_aug= np.ascontiguousarray(S_augmented, dtype=np.float64)
    d_aug = S_aug.shape[1]
    strat_sig = esig.stream2sig(S_aug, level)
    keys, cmap = generate_ito_correction_map(d_aug, d_loc, level)
    ito_sig = strat_sig.copy()
    k2i = {k:i for i,k in enumerate(keys)}
    for i, key in enumerate(keys):
        if len(key) < 2:
            continue
        for new_key, fac in cmap.get(key, []):
            j = k2i.get(new_key)
            if j is not None:
               ito_sig[i] += fac * strat_sig[j] 
    return ito_sig, strat_sig ito_sig, strat_sig
