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
"""Project configuration file

Edit the classes in this file to configure SPItorch before running one of the
targets in the Makefile.
"""

import logging

from . import logs

# =========================== Logging Parameters ==============================

class LoggingParams(logs.LP):
    """Logging parameters

    For reference, the logging levels are:

    CRITICAL (50) > ERROR (40) > WARNING (30) > INFO (20) > DEBUG (10) > NOTSET

    Logs are output for the given level and higher (e.g. logging.WARNING
    returns all warnings, errors and critical logs).
    """

    file_loc: str = './logs.txt'

    log_to_file: bool = True
    file_level: int = logging.INFO

    log_to_console: bool = True
    console_level: int = logging.INFO

    # NOTE: if set to true, you _should_ set log_to_console = False above.
    debug_logs: bool = False
    debug_level: int = logging.DEBUG
