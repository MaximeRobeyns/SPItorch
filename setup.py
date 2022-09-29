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

import re
import pathlib
from setuptools import setup
from setuptools import find_packages

wd = pathlib.Path(__file__).parent.resolve()


def get_requirements(path: str = ".") -> list[str]:
    with open(f"{path}/requirements.txt") as f:
        requirements = f.read().splitlines()
        requirements = list(filter(lambda s: "=" in s, requirements))
        return requirements


def get_deps(path: str = ".") -> list:
    deps = []
    with open(f"{path}/requirements.txt") as f:
        for line in f:
            s = re.search(r"^--find-links (?P<link>.*)$", line)
            if s is not None:
                deps.append(s.group("link"))
    return deps


deps = [d for dd in [get_deps(p) for p in (".", "docs", "tests")] for d in dd]

install_requires, docs_requires, tests_requires = map(
    get_requirements, (".", "docs", "tests")
)

with open("README.md", "r") as f:
    long_description = f.read()

with open("spt/__version__.py") as v:
    version = v.read().split('"')[-2]

setup(
    name="SPItorch",
    version=version,
    author="Maxime Robeyns",
    author_email="dev@maximerobeyns.com",
    description="Inference of Stellar Population Properties in PyTorch",
    long_description=long_description,
    url="https://github.com/MaximeRobeyns/spitorch",
    license="GPLv3",
    install_requires=install_requires,
    extras_require={
        "tests": tests_requires,
        "docs": docs_requires,
        "all": docs_requires + tests_requires,
    },
    dependency_links=deps,
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3 License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://maximerobeyns.github.io/spitorch/",
        "Bug Reports": "https://github.com/MaximeRobeyns/spitorch/issues",
        "Source": "https://github.com/MaximeRobeyns/spitorch",
    },
)
