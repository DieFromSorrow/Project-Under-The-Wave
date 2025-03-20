import requests
import random
from utils import error_print
from typing import Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class TrackCrawler:
    host = 'http://localhost:3000'
    headers = {
        'Referer': 'https://music.163.com/',
        'Host': 'music.163.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
         AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    br = 128001
    max_reconsume_times = 4

    @staticmethod
    def crawl_track(track_id):
        url = TrackCrawler.crawl_track_url(track_id)
        data = TrackCrawler.crawl_track_by_url(url)
        return data

    @staticmethod
    def crawl_track_by_url(track_url):
        if track_url is None:
            raise ValueError(f'The URL is incorrect: "{track_url}"')
        response = None
        for i in range(TrackCrawler.max_reconsume_times):
            # Use university network:
            # try:
            #     response = requests.get(url=track_url, timeout=(10, 15))
            #     response.raise_for_status()
            # except RequestException as re:
            #     error_print(f'Reconsume time: {i + 1}, error: {re}')
            #     auto_connect(wait_seconds=1)
            #     pass
            # else:
            #     break
            response = requests.get(url=track_url, timeout=(10, 15))
            response.raise_for_status()
        return response.content

    @staticmethod
    def crawl_track_url(track_ids: Union[str, list]):
        """
        通过音频 id 获取音频 url，如果传入一个 id 列表则返回对应的 url 列表
        :param track_ids: 单个音频 id 或音频 id 列表
        :return: 对应音频的 url 或 url 列表
        """
        if isinstance(track_ids, list):
            track_ids = str(track_ids)[1:-1].replace("'", '').replace(' ', '')
        url = TrackCrawler.host + f'/song/url?id={track_ids}&br={TrackCrawler.br}'
        datas = None
        for i in range(TrackCrawler.max_reconsume_times):
            # Use university network:
            # try:
            #     response = requests.get(url, timeout=(10, 15))
            #     response.raise_for_status()
            #     datas = response.json()['data']
            # except RequestException as re:
            #     error_print(f'Reconsume time: {i + 1}, error: {re}')
            #     auto_connect(wait_seconds=1)
            #     pass
            # except KeyError as ke:
            #     error_print(f'Reconsume time: {i + 1}, error: {ke}')
            #     auto_connect(wait_seconds=1)
            #     pass
            # else:
            #     break
            response = requests.get(url, timeout=(10, 15))
            response.raise_for_status()
            datas = response.json()['data']
        if datas:
            urls = [datas[i]['url'] for i in range(len(datas))]
        else:
            return None
        return urls[0] if len(urls) == 1 else urls

    @staticmethod
    def is_track_available(track_id):
        is_usable_url = TrackCrawler.host + f'/check/music?id={track_id}'
        response = requests.get(is_usable_url)
        return response.json()['success']

    # @staticmethod
    # def crawl_track_id_list(track_list_id, is_not_vip=True, duration_limit=None, batch_size=30,
    #                         limit=None, shuffle=False, use_tqdm=True, outer_tqdm: Union[tqdm, None] = None):
    #     url = TrackCrawler.host + f'/playlist/detail?id={track_list_id}'
    #     response = requests.get(url, headers=TrackCrawler.headers)
    #     response_json = response.json()
    #     try:
    #         track_list = response_json['playlist']['trackIds']
    #     except KeyError:
    #         error_print(f'crawling song list<id={track_list_id}> but the service is busy...',
    #                     end="\r")
    #         return None
    #
    #     id_list = []
    #     track_num = 0
    #     batch_list = []
    #     batch_num = 0
    #
    #     pbar = None
    #     if use_tqdm:
    #         pbar = tqdm(total=min(len(track_list), limit), desc=f'Processing id {track_list_id:<16}', ncols=100)
    #
    #     for track in track_list:
    #         if limit:
    #             if track_num >= limit:
    #                 break
    #         track_id = track['id']
    #
    #         if batch_num < batch_size:
    #             if TrackCrawler.is_track_available(track_id):
    #                 batch_list.append(track_id)
    #                 batch_num += 1
    #         else:
    #             res_list = TrackCrawler.filter(batch_list, is_not_vip, duration_limit)
    #             id_list.extend(item for item, b in zip(batch_list, res_list) if b)
    #             extend_num = sum(res_list)
    #             track_num += extend_num
    #             if pbar:
    #                 pbar.update(extend_num)
    #             batch_list = []
    #             batch_num = 0
    #
    #     if batch_num:
    #         res_list = TrackCrawler.filter(batch_list, is_not_vip, duration_limit)
    #         if batch_num == 1:
    #             res_list = [res_list]
    #         id_list.extend(item for item, b in zip(batch_list, res_list) if b)
    #         if pbar:
    #             pbar.update(sum(res_list))
    #
    #     if shuffle:
    #         random.shuffle(id_list)
    #
    #     return id_list

    @staticmethod
    def crawl_track_id_list(
        track_list_id,
        is_not_vip=True,
        duration_limit=None,
        batch_size=30,
        limit=None,
        shuffle=False,
        use_tqdm=True,
        outer_tqdm: Union[tqdm, None] = None
    ):
        url = TrackCrawler.host + f'/playlist/detail?id={track_list_id}'
        response = requests.get(url, headers=TrackCrawler.headers)
        response_json = response.json()

        try:
            track_list = response_json['playlist']['trackIds']
        except KeyError:
            error_print(f'crawling song list<id={track_list_id}> but the service is busy...', end="\r")
            return None

        id_list = []
        track_num = 0
        pbar = None

        if use_tqdm:
            total = min(len(track_list), limit) if limit else len(track_list)
            pbar = tqdm(total=total, desc=f'Processing id {track_list_id:<16}', ncols=100)

        # 预提取所有 track_id
        all_track_ids = [track['id'] for track in track_list]

        if shuffle:
            random.shuffle(all_track_ids)

        # 使用线程池批量检查 is_track_available
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有 track_id 的可用性检查任务
            future_to_id = {
                executor.submit(TrackCrawler.is_track_available, tid): tid
                for tid in all_track_ids
            }

            # 批量处理结果
            batch = []
            for future in as_completed(future_to_id):
                tid = future_to_id[future]
                try:
                    if future.result():
                        batch.append(tid)
                        # 每攒够一个批次立即过滤
                        if len(batch) >= batch_size:
                            res_list = TrackCrawler.filter(batch, is_not_vip, duration_limit)
                            id_list.extend([tid for tid, ok in zip(batch, res_list) if ok])
                            valid_count = sum(res_list)
                            track_num += valid_count
                            if pbar:
                                pbar.update(valid_count)
                            if limit and track_num >= limit:
                                break
                            batch = []
                except Exception as e:
                    error_print(f"Error checking track {tid}: {str(e)}")

        # 处理剩余未满批次
        if batch:
            res_list = TrackCrawler.filter(batch, is_not_vip, duration_limit)
            if len(batch) == 1:
                res_list = [res_list]
            id_list.extend([tid for tid, ok in zip(batch, res_list) if ok])
            valid_count = sum(res_list)
            if pbar:
                pbar.update(valid_count)

        return id_list[:limit] if limit else id_list

    @staticmethod
    def down_load_track(track_id, saving_path):
        data = TrackCrawler.crawl_track(track_id)
        with open(saving_path + str(track_id) + '.mp3', 'wb') as f:
            f.write(data)
        pass

    @staticmethod
    def filter(track_ids: Union[str, list], not_vip_track=True, duration_limit: Union[tuple, None] = None):
        res_list = [True]
        if isinstance(track_ids, list):
            res_list *= len(track_ids)
            track_ids = str(track_ids)[1:-1].replace("'", '').replace(' ', '')
        url = TrackCrawler.host + f'/song/detail?ids={track_ids}'
        response = requests.get(url)
        try:
            song_details = response.json()['songs']
        except KeyError:
            error_print(f'getting song detail<id={track_ids}> but the service is busy...')
            return None
        if not_vip_track:
            res_list = [(song_details[i]['fee'] in [0, 8]) and res_list[i] for i in range(len(song_details))]
        if duration_limit:
            res_list = [(duration_limit[0] <= song_details[i]['dt'] / 1000 <= duration_limit[1]) and res_list[i]
                        for i in range(len(song_details))]
        return res_list[0] if len(res_list) == 1 else res_list
