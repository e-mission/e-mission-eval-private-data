# This function is used for tuning
# It aims to find the best pair of trade-offs.
# - homo_second: the homogeneity score after the second round of clustering
# - percentage_second: the user labels request percentage
def get_tuning_score(homo_second,percentage_second):
    curr_score = 0.5 * homo_second + 0.5 * (1 - percentage_second)
    curr_score = float('%.3f' % curr_score)
    return curr_score
