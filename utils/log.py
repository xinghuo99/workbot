"""日志模块 - 为 workbot 提供日志记录功能

功能1: 将日志打印到控制台
功能2: 将日志存放到日志文件中
"""

import logging
import os
import time


def _setup_console_handler(logger):
    """
    功能1: 配置控制台日志输出

    :param logger: logging.Logger 实例
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


def _setup_file_handler(logger, logs_dir):
    """
    功能2: 配置文件日志输出

    日志文件命名格式: workbot_YYYYMMDD_HHMMSS.log
    存放路径: {logs_dir}/

    :param logger: logging.Logger 实例
    :param logs_dir: 日志文件存放目录
    :return: 日志文件的完整路径
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_filename = f"workbot_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return log_filepath


def setup_workbot_logger(logs_dir):
    """
    初始化 workbot 日志记录器（同时启用控制台输出和文件输出）

    :param logs_dir: 日志文件存放目录
    :return: 配置好的 logging.Logger 实例
    """
    logger = logging.getLogger("workbot")
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 功能1: 控制台输出
    _setup_console_handler(logger)

    # 功能2: 文件输出
    log_filepath = _setup_file_handler(logger, logs_dir)

    logger.info(f"日志文件: {log_filepath}")

    return logger