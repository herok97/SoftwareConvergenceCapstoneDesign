import json
import math
import os
from ortools.algorithms import pywrapknapsack_solver
import h5py
import numpy as np
import torch
from cpd_auto import cpd_auto
from cpd_nonlin import cpd_nonlin
import knapsack
# dir information
from tqdm import tqdm

base_json_dir = 'C:/Users/01079/video_summarization_data/json_files'
feature_file_dir = 'C:/Users/01079/video_summarization_data/h5_googlenet'


def simple_knapsack(values, weights, capacities):
    packed_items = []
    packed_weights = 0
    sorted_weights = [(x[1], x[2]) for x in
                      sorted(zip(values, weights, range(len(weights))), reverse=True, key=lambda x: x[0])]
    i = 0
    while 1:
        if packed_weights + sorted_weights[i][0] > capacities:
            break
        else:
            packed_items.append(sorted_weights[i][1])
            packed_weights += sorted_weights[i][0]
        i += 1
    return sorted(packed_items)


'''
    Get change points from features
        input: video feature (array)
        output: auto KTS change points (list)
'''


def get_cpt(features):
    k = np.dot(features, features.T)
    length = len(features) // 30  # in second
    m = int(math.ceil(length / 2.0))  # maximum change points, each segment is about 2s
    print(f'\nnumber of cpt: {m}')
    cps, _ = cpd_nonlin(k, m, 1)
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
    keyshot_index = simple_knapsack(avg_scores, size, capacity)
    return keyshot_index  # keyshot_index는 temporal_segments의 인덱스 범위보다 양쪽으로 1만큼 더 큰 값을 가짐 [-[ ... ]-]

def keyframe_to_keyshot(keyframes, temporal_segments, video_length):
    avg_scores = []
    size = []
    size.append(temporal_segments[0] - 0)

    # 0 번째 인덱스
    count = 0
    for keyframe in keyframes:
        if keyframe < temporal_segments[0]:
            count += 1
    avg_scores.append(count / (size[0] + 1))

    # 나머지 인덱스
    for i, cpt in enumerate(temporal_segments):
        if i > 0:
            size.append(cpt - temporal_segments[i - 1])
            count = 0
            for keyframe in keyframes:
                if keyframe < temporal_segments[0]:
                    count += 1
            avg_scores.append(count / (size[i] + 1))

    # 마지막 인덱스
    size.append(video_length - 1 - temporal_segments[-1])
    count = 0
    for keyframe in keyframes:
        if keyframe > temporal_segments[-1]:
            count += 1
    avg_scores.append(count/ (size[-1] + 1))
    capacity = round(video_length * 0.15)

    keyshot_index = simple_knapsack(avg_scores, size, capacity)
    return keyshot_index
'''
    Get feature from .h5 file
        input: feature file path
        output: feature
'''


def get_feature(path):
    with h5py.File(path, 'r') as f:
        return np.array(f['pool5'])


def get_base_json():
    video_types = ['OVP']
    data_types = ['test']
    for video_type in video_types:
        for data_type in data_types:
            # .h5 파일들에 접근
            folder_path = os.path.join(feature_file_dir, video_type, data_type)
            h5_files = os.listdir(folder_path)

            for h5_file in tqdm(h5_files, desc=video_type + '/' + data_type):
                file_path = os.path.join(folder_path, h5_file)
                # feature from .h5 file
                feature = get_feature(file_path)[:, :256]
                tqdm.write(f'{h5_file} features length: {len(feature)}')

                # get change point
                temporal_segment = get_cpt(feature).tolist()
                print(f'{h5_file}: {temporal_segment}')

                # save to json file temporal_segments: {}
                json_path = os.path.join(base_json_dir, video_type + '_json', h5_file.split('.')[0] + '.json')
                with open(json_path, 'r+') as f:
                    data = json.load(f)
                    data['temporal_segments'] = temporal_segment
                    if video_type in ['SumMe', 'TvSum']:
                        scores = data['scores']
                        keyshots = score_to_keyshot(scores, temporal_segment)
                        print(f'keyshots: {keyshots}')
                    else:
                        keyframes = data['keyframes']
                        keyshots = keyframe_to_keyshot(keyframes, temporal_segment, int(data['length']))
                        print(f'keyshots: {keyshots}')
                    data['keyshots'] = keyshots
                    f.seek(0)  # <--- should reset file position to the beginning.
                    json.dump(data, f, indent=4)
                    f.truncate()  # remove remaining part


def get_keyshots_from_scores(test_json, video_type):
    # base dir
    base_json_path = os.path.join(base_json_dir, video_type + '_json')

    # test_json 열기
    with open(test_json, 'r') as t_j:
        scores = json.load(t_j)  # 비디오 이름별 점수 가져오기

        # key shots 을 저장할 json 열기
        with open('keyshots_' + test_json, 'w') as keyshots_json:
            data = dict()
            for video_name in scores.keys():  # 각 비디오 이름에 대해
                video_scores = scores[video_name]
                video_json_path = os.path.join(base_json_path, video_name + '.json')  # 해당 비디오의 base json 파일 찾기
                with open(video_json_path, 'r') as base_json:
                    temporal_segments = json.load(base_json)['temporal_segments']
                    keyshots = score_to_keyshot(video_scores, temporal_segments)
                    print(keyshots)
                    data[video_name] = keyshots
            json.dump(data, keyshots_json, indent=4)


def evaluation(keyshot_json, video_type):
    # base dir
    base_json_path = os.path.join(base_json_dir, video_type + '_json')

    # 평가결과 평균
    total_precision = []
    total_recall = []
    total_f_score = []

    # 평가할 keyshot_json 열기
    with open(keyshot_json, 'r') as t_j:
        keyshots = json.load(t_j)  # 비디오 이름별 점수 가져오기

        for video_name in keyshots.keys():  # 각 비디오 이름에 대해
            # 평가할 keyshot
            video_keyshots = keyshots[video_name]

            video_json_path = os.path.join(base_json_path, video_name + '.json')  # 해당 비디오의 base json 파일 찾기
            # base json 으로부터 데이터 가져오기
            with open(video_json_path, 'r') as base_json:
                base_json_dict = json.load(base_json)  # 해당 비디오의 base json 파일 찾기
                temporal_segments = base_json_dict['temporal_segments']
                base_keyshots = base_json_dict['keyshots']
                video_length = base_json_dict['length']

                # 겹치는 keyshots
                overlaped_keyshots = set(video_keyshots).intersection(set(base_keyshots))

                # keyshot 의 길이 계산
                video_keyshots_duration = sum([get_duration_of_keyshot(temporal_segments, i, video_length)
                                               for i in video_keyshots])
                base_keyshots_duration = sum([get_duration_of_keyshot(temporal_segments, i, video_length)
                                              for i in base_keyshots])
                overlaped_keyshots_duration = sum([get_duration_of_keyshot(temporal_segments, i, video_length)
                                                   for i in overlaped_keyshots])

                # 겹치는 부분이 하나도 없으면 0 부여
                if overlaped_keyshots_duration == 0:
                    precision = 0
                    recall = 0
                    f_score = 0
                    pass
                else:
                    precision = overlaped_keyshots_duration / video_keyshots_duration
                    recall = overlaped_keyshots_duration / base_keyshots_duration
                    f_score = 2 * (precision * recall) / (precision + recall)

                total_precision.append(precision)
                total_recall.append(recall)
                total_f_score.append(f_score)

    print(
        f'precision: {sum(total_precision) / len(total_precision)}\nrecall: {sum(total_recall) / len(total_recall)}\nf1-score:{sum(total_f_score) / len(total_f_score)}')


def get_duration_of_keyshot(temporal_segments, i, video_length):
    if i == len(temporal_segments):  # temporal segments 의 마지막 원소면
        return int(video_length) - temporal_segments[-1]  # 그 곳의 frame num 부터 마지막 frame num 까지의 길이가 duration
    elif i == 0:
        return temporal_segments[0]  # temporal segmetns 의 첫 번째 원소면, temporal_segments[0] 의 값이 첫 번째 구간의 duration
    else:
        return temporal_segments[i] - temporal_segments[i - 1]  # 나머지의 경우 다음 인덱스와 현재 인덱스의 차이만큼이 duration


def main():
    # get_base_json()
    # get_keyshots_from_scores('OVP30.json', "OVP")
    evaluation('keyshots_OVP30.json', 'OVP')


main()
