import random
import csv
from utils import TrackCrawler


def construct_dataset(track_lists_file, train_file, test_file, limit, num_test):
    # 读取track_lists.csv文件，获取流派和歌单id的信息
    with open(track_lists_file, 'r', newline='') as file:
        reader = csv.reader(file)
        genres = next(reader)
        track_lists = list(reader)

    train_data = []
    test_data = []

    for genre_index, genre in enumerate(genres):
        print(f'\nProcessing genre name: {genre}, genre index: {genre_index} ...')
        track_count = 0
        data = set()
        enough = False
        for row in track_lists:
            track_list_id = row[genre_index]
            if not track_list_id:
                break
            track_ids = None
            while track_ids is None:
                track_ids = TrackCrawler.crawl_track_id_list(track_list_id,
                                                             is_not_vip=True,
                                                             duration_limit=(100, 500),
                                                             limit=limit - track_count + int(limit / 1.7),
                                                             shuffle=True,
                                                             use_tqdm=True, outer_tqdm=None)

            track_ids = set(track_ids)
            data = data.union(track_ids)
            track_count += len(track_ids)
            if len(data) >= limit:
                enough = True
                break

        if enough:
            data = list(data)[:limit]
            print(f'Genre {genre} is {limit} enough.')

        else:
            data = list(data)
            print(f'Warning: {genre} only has {len(data)} songs.')

        random.shuffle(data)
        train_data.append(data[:limit - num_test])
        test_data.append(data[-num_test:])

    transposed_train_data = zip(*train_data)
    transposed_test_data = zip(*test_data)

    with open(train_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(genres)
        writer.writerows(transposed_train_data)

    with open(test_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(genres)
        writer.writerows(transposed_test_data)

    print('done.')


if __name__ == '__main__':
    # 调用构造数据集函数
    _root = '../data/mini/'
    _track_lists_file = _root + 'track_lists.csv'
    _train_file = _root + 'train.csv'
    _test_file = _root + 'test.csv'
    _limit = 500  # 每个流派的歌曲id数量上限
    _num_test = 20  # 测试集的歌曲id数量

    construct_dataset(_track_lists_file, _train_file, _test_file, _limit, _num_test)
