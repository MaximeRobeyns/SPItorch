<div align="center">
<h1>SPItorch</h1>
<p>Stellar Population Inference in PyTorch</p>
</div>

SPItorch (read '_spy-torch_') is a library for estimating the
parameters of galaxies and other stellar objects.

## Installation

The [installation guide](https://maximerobeyns.github.io/SPItorch/installation.html)
contains detailed information on how to install the project, but for users
looking to get started quickly, the following steps should be sufficient.

To install, run
``` bash
git clone https://github.com/MaximeRobeyns/spitorch
cd SPItorch
make install
```

If you want to run the tutorial notebooks, you will need the tutorial datasets.
These are hosted on GitHub using [Git Large Object
Storage](https://git-lfs.github.com/) (LFS). To download it, you will need to
install `git lfs`. You can find the latest release on the
[release](https://github.com/git-lfs/git-lfs/releases) page. Here is an example
installation, using Linux:

``` bash
cd /tmp
curl -LO https://github.com/git-lfs/git-lfs/releases/download/v3.1.4/git-lfs-linux-amd64-v3.1.4.tar.gz
tar -xzf git-lfs-linux-amd64-v3.1.4.tar.gz
sudo ./install.sh
git lfs install
```

Note that we require **Python 3.9** or later. If you do not have this version,
then using a suitably configured `conda` environment is highly recommended. We
make no assumptions about your virtual environment or shell configuration,
however before calling any of the targets in the `Makefile`, please ensure that
the `python` executable in your `PATH` points to the executable/version you want
to use.

For more usage information, please see the
[documentation](https://maximerobeyns.github.io/SPItorch/) or the [tutorial notebooks](https://github.com/MaximeRobeyns/SPItorch/tree/master/tutorial_notebooks).

