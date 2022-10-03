# Create dependencies directory
if [[ ! -d ./deps ]]; then
    mkdir deps
fi

# This is required before installing fsps
export SPS_HOME=$(pwd)/deps/fsps

# Clone FSPS source code to dependencies
if [[ ! -d ./deps/fsps ]]; then
    git clone https://github.com/cconroy20/fsps.git $SPS_HOME

    # Alternative: might be better to download a stable release...
    # wget -O ./deps/fsps.tar.gz \
    #     https://github.com/cconroy20/fsps/archive/refs/tags/v3.2.tar.gz
    # tar xzf ./deps/fsps.tar.gz -C ./deps
    # mv ./deps/fsps-3.2 ./deps/fsps
    # rm ./deps/fsps.tar.gz
fi

# These flags will fix any issues arising from compiling package wheels using a
# (major) python version that differs from the current python version in your
# path.
HAMMER="--force-reinstall --ignore-installed --no-binary :all:"

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 #  $HAMMER
pip install -e .[all] --extra-index-url https://download.pytorch.org/whl/cu113
