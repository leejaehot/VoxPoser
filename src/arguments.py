"""load YAML config file"""
# YAML 형식의 config file을 로드하고, 딕셔너리처럼 동작하면서 속성방식으로도 접근할 수 있는 ConfigDict 클래스 제공 코드.
import os
import yaml

# YAML 설정 파일 로드하는 함수
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# 환경 변수(env)(rlbench) 혹은 직접 지정한 경로(config_path)를 사용해 설정 파일을 로드하는 함수
def get_config(env=None, config_path=None):
    assert env is None or config_path is None, 'env and config_path cannot be both specified'
    if config_path is None:
        assert env.lower() == 'rlbench'
        config_path = './configs/rlbench_config.yaml'
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    config = load_config(config_path)
    # wrap dict such that we can access config through attribute
    # 설정 딕셔너리를 재귀적으로 변환하는 클래스
    class ConfigDict(dict):
        def __init__(self, config):
            """recursively build config"""
            self.config = config
            for key, value in config.items():
                if isinstance(value, str) and value.lower() == 'none':
                    value = None
                if isinstance(value, dict):
                    self[key] = ConfigDict(value)
                else:
                    self[key] = value
        def __getattr__(self, key):
            return self[key]
        def __setattr__(self, key, value):
            self[key] = value
        def __delattr__(self, key):
            del self[key]
        def __getstate__(self):
            return self.config
        def __setstate__(self, state):
            self.config = state
            self.__init__(state)
    config = ConfigDict(config)
    return config

def main():
    config = get_config(config_path='./configs/rlbench_config.yaml')
    print(config)

