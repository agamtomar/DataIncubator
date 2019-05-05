
# Section 2
import itertools
import numpy as np
#
# def valid_sequence_count(n, k, j):
#     """
#     :param n: sequence of n numbers
#     :param k: range of sequence
#     :param j: Last element in sequence
#     :return:
#     """
#     if n <= 6 and k <=6:
#         # Approach 1: Brute Force Approach
#         sequences = itertools.product(range(1, k+1), repeat=n)
#
#         count = 0
#         for seq in sequences:
#             if seq[0] != 1:  # Condition for first element
#                 continue
#             elif seq[-1] != j:  # Condition for last element
#                 continue
#             elif 0 in np.diff(seq).astype('int64'):  # No two adjacent numbers are the same
#                 continue
#             else:
#                 count += 1
#                 # print(seq)
#     else:
#         # Approach 2:
#         count_wo_1 = (k-1)*((k-2)**(n-3)) # Without considering 1 to place at any location
#         count_w_1 = 0
#         for loc_1 in range(1, n-4):
#             count_w_1 += len(list(non_consecutive_combinator(list(range(n-4)), r=loc_1)))\
#                          *((k-1)**(loc_1+1))\
#                          *((k-2)**(n-2-loc_1-loc_1-1))
#
#
#         if j != 1:
#             if n % 2 == 0:
#                 count = count_wo_1+count_w_1 + 1
#             else:
#                 count = count_wo_1+count_w_1 - 1
#         else:
#             count = count_wo_1+count_w_1
#
#     return count
#
#
# def non_consecutive_combinator(rnge, r, prev=[]):
#     if r == 0:
#         yield prev
#     else:
#         for i, item in enumerate(rnge):
#             for next_comb in non_consecutive_combinator(rnge[i+2:], r-1, prev+[item]):
#                 yield next_comb

# for loc_1 in range(1, 7):
#     print(len(list(non_consecutive_combinator(list(range(7)), r=loc_1))))
#     print('\n')
# print(list(non_consecutive_combinator([1,2,3,4,5], 3)))


def valid_sequence_count(n, k, j):

    def count(n, k):
        if n == 4:
            return (k - 1) * (k - 2)
        else:
            return (k - 1) ** (n - 2) - count(n-1, k)

    j1val = count(n, k)
    if j != 1:
        if n % 2 == 0:
            return j1val + 1
        else:
            return j1val - 1
    return j1val


# 1
n = 4
k = 4
j = 2
# print(valid_sequence_count(n, k, j))
print((valid_sequence_count(n, k, j) % 10**10) + 7)

# 2
n = 4
k = 100
j = 1
print((valid_sequence_count(n, k, j) % 10**10) + 7)

# 3
n = 100
k = 100
j = 1
print((valid_sequence_count(n, k, j) % 10**10) + 7)


# 4
n = 347
k = 2281
j = 829
print((valid_sequence_count(n, k, j) % 10**10) + 7)


# 5
n = 1.26*(10**6)
k = 4.17*(10**6)
j = 1
print((valid_sequence_count(n, k, j) % 10**10) + 7)


# 6
n = 10**7
k = 10**12
j = 829
print((valid_sequence_count(n, k, j) % 10**10) + 7)
