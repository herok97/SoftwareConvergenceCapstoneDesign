import numpy as np


# dir information
json_file_dir = ''
feature_file_dir = ''

'''
    Get change points from features
        input: video feature (array)
        output: auto KTS change points (list)
'''


def get_cpt(features):
    k = np.dot(features, features.T)
    cps, _ = cpd_auto(k, len(features), 1)
    return cps

'''
    Score to Key shot
        input: Score, temporal_segments
        output: Key shot index of temporal_segments
'''
def score_to_keyshot(score, temporal_segments):
    pass


if __name__ == "__main__":
    print(1)