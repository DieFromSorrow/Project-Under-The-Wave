import requests
import random
from utils import error_print
from typing import Union
from tqdm import tqdm
from requests.exceptions import RequestException
from utils.auto_connect import auto_connect


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
        if track_url is None or track_url[-4:] != '.mp3':
            raise ValueError(f'The URL is incorrect: "{track_url}"')
        response = None
        for i in range(TrackCrawler.max_reconsume_times):
            try:
                response = requests.get(url=track_url, timeout=(10, 15))
                response.raise_for_status()
            except RequestException as re:
                error_print(f'Reconsume time: {i + 1}, error: {re}')
                auto_connect(wait_seconds=1)
                pass
            else:
                break
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
            try:
                response = requests.get(url, timeout=(10, 15))
                response.raise_for_status()
                datas = response.json()['data']
            except RequestException as re:
                error_print(f'Reconsume time: {i + 1}, error: {re}')
                auto_connect(wait_seconds=1)
                pass
            except KeyError as ke:
                error_print(f'Reconsume time: {i + 1}, error: {ke}')
                auto_connect(wait_seconds=1)
                pass
            else:
                break
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

    @staticmethod
    def crawl_track_id_list(track_list_id, is_not_vip=True, duration_limit=None,
                            limit=None, shuffle=False, use_tqdm=True, outer_tqdm: Union[tqdm, None] = None):
        url = TrackCrawler.host + f'/playlist/detail?id={track_list_id}'
        response = requests.get(url, headers=TrackCrawler.headers)
        response_json = response.json()
        try:
            track_list = response_json['playlist']['trackIds']
        except KeyError:
            error_print(f'crawling song list<id={track_list_id}> but the service is busy...',
                        end="\r")
            return None

        id_list = []
        track_num = 0
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=min(len(track_list), limit), desc=f'Processing id {track_list_id:<16}', ncols=100)

        for track in track_list:
            if limit:
                if track_num >= limit:
                    break
            track_id = track['id']
            if TrackCrawler.is_track_available(track_id) and TrackCrawler.filter(track_id, is_not_vip, duration_limit):
                id_list.append(track_id)
                track_num += 1
                if pbar:
                    pbar.update(1)

        if shuffle:
            random.shuffle(id_list)

        return id_list

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
