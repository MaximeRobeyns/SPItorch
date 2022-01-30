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
"""Project-wide utilities file"""

from typing import Optional

from . import logs
from .config import LoggingParams

def configure_logging(logging_params: logs.LP = LoggingParams(),
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
    dictConfig(logs.get_logging_config(logging_params))
    # logging.info(
    #     f'\n\n{79*"~"}\n\n\tAGNFinder\n\t{time.ctime()}\n\n{79*"~"}\n\n')
