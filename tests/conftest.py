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
"""Configuration for PyTest"""


import logging
import pytest


from spt.logs import configure_logging
from spt.config import LoggingParams


class LP(LoggingParams):
    log_to_file: bool = True
    file_level: int = logging.WARNING
    file_loc: str = './tests/logs.txt'

    log_to_console: bool = True
    console_level: int = logging.WARNING

    debug_logs: bool = False


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    # configure_logging(file=True, file_loc='./tests/logs.txt')
    configure_logging(LP())


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in the cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
