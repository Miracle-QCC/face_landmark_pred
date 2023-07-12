# _*_ coding: utf-8 _*_
import logging
import time

import colorlog
from colorlog import ColoredFormatter


class Logger(object):
    def __init__(self,  save_path='.', logger=__name__, level=logging.INFO):
        self.__name = logger
        self.__path = save_path + "/" +  time.strftime('%Y%m%d-%H:%M:%S', time.localtime(time.time())) + '.log'
        self.__level = level
        self.__logger = logging.getLogger(self.__name)
        self.__logger.setLevel(self.__level)

    def __ini_handler(self):
        """初始化handler"""
        stream_handler = colorlog.StreamHandler()
        file_handler = logging.FileHandler(self.__path, encoding='utf-8')
        return stream_handler, file_handler

    def __set_handler(self, stream_handler, file_handler, level='DEBUG'):
        """设置handler级别并添加到logger收集器"""
        stream_handler.setLevel(level)
        file_handler.setLevel(level)
        self.__logger.addHandler(stream_handler)
        self.__logger.addHandler(file_handler)

    def __set_formatter(self, stream_handler, file_handler):
        """设置日志输出格式"""
        # formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s-%(filename)s-[line:%(lineno)d]'
        #                               ': %(message)s',
        #                               # datefmt='%Y-%m-%d,%H:%M:%S.%f'
        #                               )

        formatter = logging.Formatter('[%(levelname)s] %(asctime)s'
                                      ': %(message)s',
                                      # datefmt='%Y-%m-%d,%H:%M:%S.%f'
                                      )
        formatter_console = ColoredFormatter(
            '%(log_color)s[%(levelname)s] %(asctime)s : %(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'bold_yellow',
                'ERROR': 'bold_red',
                'CRITICAL': 'bold_red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
        stream_handler.setFormatter(formatter_console)
        file_handler.setFormatter(formatter)

    def __close_handler(self, stream_handler, file_handler):
        """关闭handler"""
        stream_handler.close()
        file_handler.close()

    # @property
    # def Logger(self):\
    def getlog(self):
        """构造收集器，返回looger"""
        stream_handler, file_handler = self.__ini_handler()
        self.__set_handler(stream_handler, file_handler)
        self.__set_formatter(stream_handler, file_handler)
        self.__close_handler(stream_handler, file_handler)
        return self.__logger

if __name__ == '__main__':
    # save_path表示要存放日志的位置
    logger = Logger(logger="testlogger", save_path="").getlog()

    logger.info('info')
    logger.error('err')
    logger.warning('war')
    logger.debug('debug')
    logger.critical('cti')

