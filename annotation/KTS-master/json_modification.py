import json
import os
from ortools.algorithms import pywrapknapsack_solver
import h5py
import numpy as np
import torch
from cpd_auto import cpd_auto
import knapsack
# dir information
from tqdm import tqdm

json_file_dir = 'C:/Users/01079/video_summarization_data/json_files'
feature_file_dir = 'C:/Users/01079/video_summarization_data/h5_googlenet'
solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'Knapsack')


def knapsack(values, weights, capacities):
    solver.Init(values, weights, capacities)
    computed_value = solver.Solve()
    packed_items = []
    packed_weights = []
    total_weight = 0
    print('Total value =', computed_value)
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            total_weight += weights[0][i]
    print('Total weight:', total_weight)
    print('Packed items:', packed_items)
    print('Packed_weights:', packed_weights)
    return packed_items


'''
    Get change points from features
        input: video feature (array)
        output: auto KTS change points (list)
'''


def get_cpt(features):
    k = np.dot(features, features.T)
    print(f'\nMax number of cpt: {50}')
    # cps, _ = cpd_auto(k, len(features) // 60, 1)
    cps, _ = cpd_auto(k, 50, 1)
    return cps


'''
    Score to Key shot
        input: Score, temporal_segments
        output: Key shot index of temporal_segments
'''


def score_to_keyshot(scores, temporal_segments):
    # sum scores
    avg_scores = []
    size = []
    size.append(temporal_segments[0] - 0)
    avg_scores.append(sum(scores[0:temporal_segments[0]]) / (size[0] + 1))
    for i, cpt in enumerate(temporal_segments):
        if i > 0:
            size.append(cpt - temporal_segments[i - 1])
            avg_scores.append(sum(scores[temporal_segments[i - 1]:cpt]) / (size[i] + 1))
    size.append(len(scores) - 1 - temporal_segments[-1])
    avg_scores.append(sum(scores[temporal_segments[-1]:]) / (size[-1] + 1))

    capacity = round(len(scores) * 0.15)
    keyshot_index = knapsack(avg_scores, [size], [capacity])

    return keyshot_index  # keyshot_index는 temporal_segments의 인덱스 범위보다 양쪽으로 1만큼 더 큰 값을 가짐 [-[ ... ]-]


'''
    Get feature from .h5 file
        input: feature file path
        output: feature
'''


def get_feature(path):
    with h5py.File(path, 'r') as f:
        return np.array(f['pool5'])


if __name__ == "__main__":
    video_types = ['SumMe', 'TvSum', 'OVP']
    data_types = ['train', 'test']

    for video_type in video_types:
        for data_type in data_types:
            # .h5 파일들에 접근
            folder_path = os.path.join(feature_file_dir, video_type, data_type)
            h5_files = os.listdir(folder_path)

            for h5_file in tqdm(h5_files, desc=video_type + '/' + data_type):
                file_path = os.path.join(folder_path, h5_file)
                # feature from .h5 file
                feature = get_feature(file_path)
                tqdm.write(f'{h5_file} features length: {len(feature)}')

                # get change point
                temporal_segment = get_cpt(feature).tolist()
                print(f'{h5_file}: {temporal_segment}')

                # save to json file temporal_segments: {}
                json_path = os.path.join(json_file_dir, video_type + '_json', h5_file.split('.')[0] + '.json')
                with open(json_path, 'r+') as f:
                    data = json.load(f)
                    data['temporal_segments'] = temporal_segment
                    if video_type in ['SumMe', 'TvSum']:
                        scores = data['scores']
                        keyshots = score_to_keyshot(scores, temporal_segment)
                        print(f'keyshots: {keyshots}')
                    data['keyshots'] = keyshots
                    f.seek(0)  # <--- should reset file position to the beginning.
                    json.dump(data, f, indent=4)
                    f.truncate()  # remove remaining part
