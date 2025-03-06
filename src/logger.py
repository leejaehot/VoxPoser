import logging
import sys
import re

class Logger:
    def __init__(self, filename="log_CaP/log_output.txt", mode="w"):
        """
        jupyter 코드 출력을 파일과 터미널에 동시에 기록하는 Logger 클래스
        :param filename: 로그 파일 이름 (기본값: log_output.txt)
        :param mode: 파일 열기 모드 (w덮어쓰기, a이어쓰기)
        """

        self.terminal = sys.stdout
        self.log = open(filename, mode)
        self.filename = filename
        sys.stdout = self
    
    def write(self, message):
        clean_message = self._remove_ansi_escape(message) # ANSI escape code remove
        self.terminal.write(clean_message)
        self.terminal.flush()
        self.log.write(clean_message)
        self.log.flush()
    
    def flush(self):
        pass
    
    def close(self):
        """
        로그 파일을 닫고 표준출력을 원래 상태로 복구
        """
        sys.stdout = self.terminal
        self.log.close()
    def _remove_ansi_escape(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)  # ANSI 코드 제거 후 반환


def setup_logging(log_filename="log_CaP/log_output.txt", level=logging.INFO):

    logging.basicConfig(
        filename=log_filename,
        level=level,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )