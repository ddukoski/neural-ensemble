import logging
import logging.handlers
import os
from colorama import Fore, Back, Style, init

init(autoreset=True)


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Back.RED + Fore.WHITE + Style.BRIGHT,
    }
    MSG_COLORS = {logging.INFO: Fore.LIGHTBLACK_EX}

    def format(self, record):
        log_fmt = "\t[%(levelname)s] [%(asctime)s]: %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
        base_msg = formatter.format(record)

        level_color = self.COLORS.get(record.levelno, "")
        msg_color = self.MSG_COLORS.get(record.levelno, "")

        colored_level = f"{level_color}{record.levelname}{Style.RESET_ALL}"

        base_msg = base_msg.replace(f"[{record.levelname}]", f"[{colored_level}]", 1)
        msg_start = base_msg.find("]: ") + 3
        return (
            f"{base_msg[:msg_start]}{msg_color}{base_msg[msg_start:]}{Style.RESET_ALL}"
        )


def make_logger(name: str = __name__, log_file: str = "builds.log") -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        return logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter())
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(levelname)s] [%(asctime)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
