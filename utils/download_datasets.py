from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from datasets.mfcc_dataset import MfccDatesetOnline


class DatasetPreprocessor:
    def __init__(self, src_dataset, save_dir, max_workers=8):
        self.src_dataset = src_dataset
        self.save_dir = Path(save_dir)
        self.mfcc_dir = self.save_dir
        self.mfcc_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _process_item(self, index):
        try:
            # 显式使用with torch.no_grad()
            with torch.no_grad():
                mfcc, label = self.src_dataset[index]

                # 使用del和显存释放
                cpu_mfcc = mfcc.cpu().clone()
                del mfcc  # 删除GPU引用
                torch.cuda.empty_cache()  # 立即释放未使用显存

                filename = f"{index:04d}.pt"
                save_path = self.mfcc_dir / filename
                torch.save((cpu_mfcc, label), save_path)

                # 清理CPU内存
                del cpu_mfcc
                return filename, label
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            return None

    def save_all(self):
        futures = []
        for idx in range(len(self.src_dataset)):
            futures.append(self.executor.submit(self._process_item, idx))

        # 进度条监控
        with tqdm(total=len(futures)) as pbar:
            for future in futures:
                result = future.result()
                if result:
                    self.metadata.append(result)
                pbar.update(1)

        # 保存元数据
        meta_df = pd.DataFrame(self.metadata, columns=["filename", "label"])
        meta_df.to_csv(self.save_dir / "metadata.csv", index=False)

    def repair(self, err_index_list: list):
        for index in err_index_list:
            self._process_item(index)


def save_all(data_folder: str, saved_folder: str, train: bool):
    if train:
        filename = 'train'
    else:
        filename = 'test'
    online_dataset = MfccDatesetOnline(f"../data/{data_folder}/{filename}.csv", cuda=True)
    preprocessor = DatasetPreprocessor(online_dataset, f"../data/{saved_folder}/{filename}", max_workers=8)
    preprocessor.save_all()


def repair(index_list):
    online_dataset = MfccDatesetOnline("../data/mini/train.csv", cuda=True)
    preprocessor = DatasetPreprocessor(online_dataset, "../data/mini_saved_mfcc/train", max_workers=8)
    preprocessor.repair(index_list)


if __name__ == '__main__':
    repair([13696])
