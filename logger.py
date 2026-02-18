import sys
import logging
from config import Config

try:
    from loguru import logger
    USE_LOGURU = True
except ImportError:
    USE_LOGURU = False

def setup_logger():
    if USE_LOGURU:
        logger.remove()
        
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=Config.LOG_LEVEL
        )
        
        logger.add(
            Config.LOG_DIR / "financial_tool_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=Config.LOG_LEVEL
        )
        
        return logger
    else:
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(Config.LOG_DIR / "financial_tool.log")
            ]
        )
        
        class SimpleLogger:
            def __init__(self):
                self.logger = logging.getLogger(__name__)
            
            def info(self, msg): self.logger.info(msg)
            def warning(self, msg): self.logger.warning(msg)
            def error(self, msg): self.logger.error(msg)
            def debug(self, msg): self.logger.debug(msg)
        
        return SimpleLogger()

log = setup_logger()
