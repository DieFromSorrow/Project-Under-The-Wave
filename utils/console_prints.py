import torch
import logging
import time
from pathlib import Path
# from colorama import init, Fore


# current_time = time.strftime("%m-%d-%H", time.localtime())
# log_file_path = f"../logs/console_logs/{current_time}.log"
# log_file_path = Path(log_file_path)
#
# # 检查日志文件是否已存在
# if not log_file_path.exists():
#     # 创建日志文件
#     log_file_path.touch()
#
# # 配置日志记录器
# logging.basicConfig(filename=log_file_path, level=logging.WARNING)


def error_print(e_str, **kwargs):
    print('UTW-Error:', e_str, **kwargs)
    logging.error(e_str)


def running_print(*r_strs):
    print('UTW-Running:', *r_strs)
    logging.info(r_strs)


def processing_print(*p_strs):
    print('UTW-Processing:', *p_strs)
    logging.info(p_strs)


def warning_print(*w_strs):
    print('UTW-Warning:', *w_strs)
    logging.warning(w_strs)


def targets_print(t: torch.Tensor):
    running_print('Targets:', t.tolist())


def outputs_print(o: torch.Tensor):
    running_print('Outputs:', o.tolist())
