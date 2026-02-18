import os
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config:
    BASE_DIR = Path(__file__).parent
    
    DATA_DIR = BASE_DIR / os.getenv('DATA_DIR', 'data')
    LOG_DIR = BASE_DIR / os.getenv('LOG_DIR', 'logs')
    DATA_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')
    
    DB_TYPE = os.getenv('DB_TYPE', 'sqlite')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 5432))
    DB_NAME = os.getenv('DB_NAME', 'financial_tool')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
    
    if DB_TYPE == 'sqlite':
        DATABASE_URL = f'sqlite:///{DATA_DIR / "financial.db"}'
    elif DB_TYPE == 'postgresql':
        DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    elif DB_TYPE == 'mysql':
        DATABASE_URL = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


class UserConfig:
    DEFAULT_CONFIG = {
        'pool_type': 'core',
        'risk_level': 'low',
        'stock_count': 80,
        'max_stock_count': 80,
        'short_term_top_n': 5,
        'medium_term_top_n': 5,
        'long_term_top_n': 5,
        'data_days': 365,
        'cache_days': 1
    }
    
    def __init__(self, config_file: Path = None):
        if config_file is None:
            config_file = Config.BASE_DIR / 'user_config.json'
        self.config_file = config_file
        self._config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    self._config.update(saved_config)
            except Exception as e:
                print(f"警告: 加载用户配置失败，使用默认配置: {e}")
    
    def save(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"警告: 保存用户配置失败: {e}")
    
    def get(self, key: str, default=None):
        return self._config.get(key, default)
    
    def set(self, key: str, value, auto_save: bool = True):
        self._config[key] = value
        if auto_save:
            self.save()
    
    def get_all(self) -> dict:
        return self._config.copy()
    
    @property
    def pool_type(self) -> str:
        return self._config.get('pool_type', 'core')
    
    @pool_type.setter
    def pool_type(self, value: str):
        self.set('pool_type', value)
    
    @property
    def risk_level(self) -> str:
        return self._config.get('risk_level', 'low')
    
    @risk_level.setter
    def risk_level(self, value: str):
        self.set('risk_level', value)
    
    @property
    def stock_count(self) -> int:
        return self._config.get('stock_count', 80)
    
    @stock_count.setter
    def stock_count(self, value: int):
        self.set('stock_count', value)
    
    @property
    def short_term_top_n(self) -> int:
        return self._config.get('short_term_top_n', 5)
    
    @short_term_top_n.setter
    def short_term_top_n(self, value: int):
        self.set('short_term_top_n', value)
    
    @property
    def medium_term_top_n(self) -> int:
        return self._config.get('medium_term_top_n', 5)
    
    @medium_term_top_n.setter
    def medium_term_top_n(self, value: int):
        self.set('medium_term_top_n', value)
    
    @property
    def long_term_top_n(self) -> int:
        return self._config.get('long_term_top_n', 5)
    
    @long_term_top_n.setter
    def long_term_top_n(self, value: int):
        self.set('long_term_top_n', value)
    
    @property
    def data_days(self) -> int:
        return self._config.get('data_days', 300)
    
    @data_days.setter
    def data_days(self, value: int):
        self.set('data_days', value)
    
    @property
    def cache_days(self) -> int:
        return self._config.get('cache_days', 1)
    
    @cache_days.setter
    def cache_days(self, value: int):
        self.set('cache_days', value)
    
    @property
    def max_stock_count(self) -> int:
        return self._config.get('max_stock_count', 80)
    
    @max_stock_count.setter
    def max_stock_count(self, value: int):
        self.set('max_stock_count', value)


user_config = UserConfig()
