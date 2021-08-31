
import random
import numpy as np
from collections import defaultdict
from functools import reduce

"""
==================
Resample code
==================
notes:
set(y_array) # which class
{0, 1}
set(group_array) # which subgroup
{0, 1, 2, 3}
set(split_array) # which training set
{0, 1, 2}
len(y_array) = len(group_array) = len(split_array)
"""

def multiple_logical_or(lst):
    return reduce(np.logical_or, lst)

def get_arr(split_idx_lst, group_idx_lst, split_array, group_array):
    split_arr_subset = multiple_logical_or([split_array == split_idx for split_idx in split_idx_lst])
    group_arr_subset = multiple_logical_or([group_array == group_idx for group_idx in group_idx_lst])
    return np.logical_and(split_arr_subset, group_arr_subset)

def get_arr_single(split_idx, group_idx, split_array, group_array):
    return get_arr([split_idx], [group_idx], split_array, group_array)

def get_counts(split_array, group_array):
    """ Get subgroup counts for each split group. """
    dct = defaultdict(lambda: defaultdict(dict))
    for split_idx in list(set(split_array)):
        for group_idx in list(set(group_array)):
            dct[split_idx][group_idx] = sum(get_arr_single(split_idx, group_idx, split_array, group_array))
    return dct

def resample(split_array, group_array, counts):
    """ Resamples within each subgroup."""
    # split_array = split_array.copy()
    splits = list(set(split_array))

    split_array = np.array([-1] * len(group_array))
    groups = list(set(group_array))

    for group_idx in groups:
        label_mask = group_array == group_idx
        indices = np.where(label_mask)[0]
        random.shuffle(indices)

        start_idx, end_idx = 0, 0
        for split_idx in splits:
            start_idx = end_idx
            end_idx += counts[split_idx][group_idx]

            # we set a bunch of indices in that group to a split index (val/test/train split)
            split_array[indices[start_idx:end_idx]] = split_idx
    return split_array

