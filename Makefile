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

SHELL := bash
# .ONESHELL:
# .SHELLFLAGS := -eu -o pipefail -c
# .DELETE_ON_ERROR:
# MAKEFLAGS += --warn-undefined-variables
# MAKEFLAGS += --no-builtin-rules
# MAKEFLAGS += -j8

# Ensure Python >= 3.9 is present ---------------------------------------------

# Specify the path to your Python>=3.9 executable here.
PYTHON = $(shell which python3.9)

PYTHON_VERSION_MIN=3.9
PYTHON_VERSION=$(shell $(PYTHON) -c 'import sys; print("%d.%d"% sys.version_info[0:2])' )
PYTHON_VERSION_OK=$(shell $(PYTHON) -c 'import sys;\
  print(int(float("%d.%d"% sys.version_info[0:2]) >= $(PYTHON_VERSION_MIN)))' )

ifeq ($(PYTHON_VERSION_OK),0)
  $(error "Need python $(PYTHON_VERSION) >= $(PYTHON_VERSION_MIN)")
endif

# Targets ---------------------------------------------------------------------

default: test

dset: ## To create an offline dataset of theta-photometry samples.
	@python spt/modelling/simulation.py

mdset:  ## To create an offline dataset, using mpi
	@$(eval N := $(shell python -c\
	'from spt.config import SamplingParams as S; print(S().concurrency)'))
	mpirun -n $N python spt/modelling/mpi_simulator.py

mypy: ## To run mypy only (this is usually done with test / alltest)
	@mypy

test: mypy  ## To run the program's fast tests (e.g. to verify an installation)
	@python -m pytest -s tests

alltest: mypy ## To run all the program's tests (including slow running ones)
	@python -m pytest tests --runslow

docs:  ## Start compiling the documentation and watching for changes
	@./docs/writedocs.sh

.PHONY: install
install: ## To install everything (requires internet connection)
	@./bin/install.sh

kernel:  ## To setup a Jupyter kernel to run notebooks in SPItorch virtual env
	python -m ipykernel install --user --name agnvenv \
		--display-name "SPItorch (Python 3.10)"

lab: ## To start a Jupyter Lab server
	@export SPS_HOME=$(shell pwd)/deps/fsps
	jupyter lab --notebook-dir=. --ip=0.0.0.0  --port 8881 # --collaborative --no-browser

san: ## To run the SAN inference code specifically
	@export SPS_HOME=$(shell pwd)/deps/fsps
	@python spt/inference/san.py

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: dset mdset mypy test alltest kernel lab san help docs
