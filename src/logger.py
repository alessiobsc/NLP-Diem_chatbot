import logging
import sys
from logging.handlers import RotatingFileHandler
from config import LOG_LEVEL, LOG_DIR, LOG_FILE, MAX_LOG_SIZE_MB, LOG_BACKUP_COUNT

def get_logger(name: str) -> logging.Logger:
    """
    This function returns a logger instance configured to log to the console
    and a rotating file handler to prevent infinite growth.
    
    The log level is determined by the LOG_LEVEL configuration variable.

    Args:
        name: The name of the logger.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Use LOG_LEVEL from config, which already handles the .env and default
        log_level_str = LOG_LEVEL.upper()
        
        # Map string to logging level, fallback to INFO if invalid
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        logger.setLevel(log_level)
        
        # Formatter used by all handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 1. Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File Handler with Rotation (to prevent memory/disk overflow)
        # Create log directory if it doesn't exist
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Max size in bytes (MB * 1024 * 1024)
        max_bytes = MAX_LOG_SIZE_MB * 1024 * 1024
        
        file_handler = RotatingFileHandler(
            filename=LOG_FILE,
            maxBytes=max_bytes,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
