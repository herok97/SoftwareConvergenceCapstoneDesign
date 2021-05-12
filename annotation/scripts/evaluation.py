
# 평가된 score json file, video_type =>
import json
import os

base_json_dir = 'C:/Users/01079/video_summarization_data/json_files'

def get_keyshots_from_scores(test_json, video_type):
    # base dir
    base_json_path = os.path.join(base_json_dir, video_type, 'test')

    # test_json 열기
    with open(test_json, 'r') as t_j:
        scores = json.load(t_j)     # 비디오 이름별 점수 가져오기

        # key shots 을 저장할 json 열기
        with open('keyshots_'+test_json, 'w') as keyshots_json:
            data = dict()
            for video_name in scores.keys():    # 각 비디오 이름에 대해
                video_scores = scores[video_name]
                video_json_path = os.path.join(base_json_path, video_name+'.json')  # 해당 비디오의 base json 파일 찾기
                with open(video_json_path, 'r') as base_json:
                    temporal_segments = json.load(base_json)['temporal_segments']
                    keyshots = scores_to_keyshots(video_scores, temporal_segments)
                    data[video_name] = keyshots
            json.dump(data, keyshots_json, indent=4)



    pass


def evaluation():
    pass


if __name__ == "__main__":
    get_keyshots_from_scores('TvSum_epoch-50.json', "TvSum")