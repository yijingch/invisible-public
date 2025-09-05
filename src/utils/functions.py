import numpy as np 
from typing import List, Any, Dict
from sklearn.metrics.pairwise import cosine_similarity


def get_p(arr_bstr, emp):
    numer = len([x for x in arr_bstr if x > emp])
    denom = len(arr_bstr)
    return numer/denom


def normalize(arr, smooth=0):
    if np.sum(arr) > 0:
        return (np.array(arr)+smooth)/np.sum(np.array(arr)+smooth)
    else:
        arr


def get_cosine_similarity(vec1, vec2) -> float:
    return cosine_similarity(np.array(vec1).reshape(1,-1), np.array(vec2).reshape(1,-1))[0][0]


def get_jaccard_similarity(set1, set2) -> float:
    union = set(set1).union(set(set2))
    inter = set(set1).intersection(set(set2))
    return len(inter)/len(union)


def compare_distribution_with_sig(
        sample1:np.array, 
        sample2:np.array, 
        test_func:Any,
        bruns:str=100,
        sample_size=-1,
        print_info=True):

    if sample_size > 0:
        N1 = N2 = sample_size 
    else:
        N1 = len(sample1)
        N2 = len(sample2)

    emp_test = test_func(sample1, sample2)

    assert N1 > 0 and N2 > 0, "please input non-emtpy samples!"

    # bootstrap
    bstr_test1 = np.zeros(bruns)
    bstr_test2 = np.zeros(bruns)
    for r in range(bruns):
        bstr_sample1 = np.random.choice(sample1, N1, replace=True)
        bstr_sample2 = np.random.choice(sample2, N2, replace=True)
    
        test_bstr1 = test_func(sample1, bstr_sample1)
        test_bstr2 = test_func(sample2, bstr_sample2)

        bstr_test1[r] = test_bstr1[0]
        bstr_test2[r] = test_bstr2[0]

    p1 = get_p(bstr_test1, emp_test[0])
    p2 = get_p(bstr_test2, emp_test[0])
    if print_info: print("N1 =", N1, "\tN2 =", N2)
    return p1, p2


def compare_distribution_with_sig_weighted(
        sample1:np.array, 
        sample2:np.array, 
        weights1:np.array,
        weights2:np.array,
        test_func,
        bruns:str=100,
        sample_size=-1):

    if sample_size > 0:
        N1 = N2 = sample_size 
    else:
        N1 = len(sample1)
        N2 = len(sample2)

    assert N1 > 0 and N2 > 0, "please input non-emtpy samples!"

    res_sample1 = np.random.choice(sample1, len(sample1), p=weights1, replace=True)
    res_sample2 = np.random.choice(sample2, len(sample2), p=weights2, replace=True)
    emp_test = test_func(res_sample1, res_sample2)

    # bootstrap
    bstr_test1 = np.zeros(bruns)
    bstr_test2 = np.zeros(bruns)
    for r in range(bruns):
        bstr_sample1 = np.random.choice(sample1, N1, p=weights1, replace=True)
        bstr_sample2 = np.random.choice(sample2, N2, p=weights2, replace=True)
    
        test_bstr1 = test_func(res_sample1, bstr_sample1)
        test_bstr2 = test_func(res_sample2, bstr_sample2)

        bstr_test1[r] = test_bstr1[0]
        bstr_test2[r] = test_bstr2[0]

    p1 = get_p(bstr_test1, emp_test[0])
    p2 = get_p(bstr_test2, emp_test[0])
    print("N1 =", N1, "\tN2 =", N2)
    return p1, p2