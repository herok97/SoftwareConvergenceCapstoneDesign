from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import json
from sklearn.preprocessing import MinMaxScaler

def minMaxScaling(data):
    m = min(data)
    M = max(data)
    data = [(x-m)/(M-m) for x in data]
    return data

def get_lenUserScore(video_type, score_path, video_name):
    if video_type is 'SumMe':
        score = io.loadmat(f'{score_path}/{video_name}.mat')
        length = len(score['user_score'])
        return length

def get_meanUserScore(video_type, score_path, video_name):
    if video_type is 'SumMe':
        score = io.loadmat(f'{score_path}/{video_name}.mat')
        scores = [np.mean(row) for row in score['user_score']]
        return ss.zscore(scores)

def get_myScore(model_path, video_name):
    with open(model_path, 'r') as j:
        data = json.load(j)
    return ss.zscore(data[f'{video_name}.h5'])

def get_topScoreIndex(score, rate):
    l = len(score)
    topid = sorted(range(l), key=lambda i: score[i])[-int(rate*l):]
    return topid

def cal_precision_recall_F(my_score, user_score, rate):

    # top rate index
    topScoreIndex_my = get_topScoreIndex(my_score, rate)
    topScoreIndex_user = get_topScoreIndex(user_score, rate)

    duration_my = len(topScoreIndex_my)
    duration_user = len(topScoreIndex_user)
    count = 0
    for i in topScoreIndex_user:
        if i in topScoreIndex_my:
            count += 1

    P = round(count / duration_my, 3)
    R = round(count / duration_user, 3)
    F = round(2 * P * R / (P + R) * 100, 3)

    return P, R, F

# video_type = 'SumMe'
# score_path = './GT'
# video_name = 'Excavators river crossing'
# model_path = 'SumMe_25.TvSum_json'
# myscore = get_myScore(model_path, video_name)
# userscore = get_meanUserScore(video_type, score_path, video_name)
#
# print(cal_precision_recall_F(myscore, userscore, 0.3))
# plt.plot(userscore)
# plt.plot(myscore)
# plt.title('video: Excavators river crossing')
# plt.legend(['userscore', 'myscore'])
# plt.axis([0,10000,-5,5])
# plt.show()



video_type = 'SumMe'
score_path = './GT'
video_name = 'Notre_Dame'
scoreLen = get_lenUserScore(video_type, score_path, video_name)
print(scoreLen)