# SPItorch: Inference of Stellar Population Properties in PyTorch
#
# Copyright (C) 2022 Maxime Robeyns <dev@maximerobeyns.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
"""Logging configuration.

Note that we use the Rich logging interface. Among other things, you can output
a traceback after an exception with logging.exception("something bad happened").
"""

from typing import Any, Optional

from .config import LoggingParams

# from abc import ABCMeta, abstractproperty
# class LP(metaclass=ABCMeta):
#     """Abstract base parameters"""
#
#     # TODO: devise a way to (simply & succinctly) define properties and their
#     # setters / deleters...
#     @abstractproperty
#     def file_loc(self) -> str:
#         return './logs.txt'
#     @file_loc.setter
#     def file_loc(self, val: str):
#         self.file_loc = val
#
#     @abstractproperty
#     def log_to_file(self) -> bool:
#         return True
#     @log_to_file.setter
#     def log_to_file(self, val: bool):
#         self.log_to_file = val
#
#     @abstractproperty
#     def file_level(self) -> int:
#         return logging.INFO
#
#     @abstractproperty
#     def log_to_console(self) -> bool:
#         return True
#     @log_to_console.setter
#     def log_to_console(self, val: bool):
#         self.log_to_console = val
#
#     @abstractproperty
#     def console_level(self) -> int:
#         return logging.INFO
#
#     @abstractproperty
#     def debug_logs(self) -> bool:
#         return False
#     @debug_logs.setter
#     def debug_logs(self, val: bool):
#         self.debug_logs = val
#
#     @abstractproperty
#     def debug_level(self) -> int:
#         return logging.DEBUG


def get_logging_config(p: LoggingParams = LoggingParams()) -> dict[str, Any]:

    handlers = []
    if p.log_to_file:
        handlers.append('file')
    if p.debug_logs:
        handlers.append('console_debug')
    if p.log_to_console:
        handlers.append('console')

    return {
        'version': 1,
        'disable_existing_loggers': False,
        # https://docs.python.org/3/library/logging.html#logrecord-attributes
        'formatters': {
            'standard': {
                'format': '[%(asctime)s %(levelname)s] %(message)s'
            },
            'debug': {
                'format': '%(message)s (in %(filename)s:%(lineno)d)'
            },
        },
        'handlers': {
            # only log errors out to the console
            'console': {
                'class': 'rich.logging.RichHandler',
                'level': p.console_level,
                'show_time': False,
                'omit_repeated_times': True,
                'show_level': True,
                'show_path': True,
                'enable_link_path': True,
                'markup': True,
                'rich_tracebacks': False,
                'log_time_format': "[%x %X]",
            },
            'console_debug': {
                'class': 'rich.logging.RichHandler',
                'formatter': 'debug',
                'level': p.debug_level,
                'show_time': True,
                'omit_repeated_times': True,
                'show_level': True,
                'show_path': True,
                'enable_link_path': True,
                'markup': True,
                'rich_tracebacks': True,
                'tracebacks_show_locals': True,
                'log_time_format': "[%X]",
            },
            # Output more information to a file for post-hoc analysis
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'level': p.file_level,
                'filename': p.file_loc,
                'mode': 'a',
                'encoding': 'utf-8',
                'maxBytes': 500000,
                'backupCount': 4
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': handlers,
                'level': 'NOTSET',
                'propagate': True
            }
        }
    }


def configure_logging(logging_params: LoggingParams = LoggingParams(),
                      debug: Optional[bool] = None,
                      file: Optional[bool] = None,
                      file_loc: Optional[str] = None) -> None:
    """Performs a one-time configuration of the root logger for the program.

    Note that all the arguments are optional, and if omitted the default values
    in config.py will be used.

    Args:
        logging_params: An instance of spt.logs.LP to customise logs outside
            config.py
        debug: override to output debug logs only if True; console logs only if
            False
        file: override to enable (True) or disable (False) file logging.
        file_loc: location of the log file

    Example:
        >>> configure_logging(debug=True, file_loc='/tmp/debug.log')
    """
    if debug is not None:
        logging_params.log_to_console = not debug
        logging_params.debug_logs = not debug
    if file is not None:
        logging_params.log_to_file = file

    if file_loc is not None:
        logging_params.file_loc = file_loc

    from logging.config import dictConfig
    dictConfig(get_logging_config(logging_params))
    # logging.info(
    #     f'\n\n{79*"~"}\n\n\tAGNFinder\n\t{time.ctime()}\n\n{79*"~"}\n\n')
