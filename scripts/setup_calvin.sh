pip install open-flamingo==2.0.1
pip install lightning==2.3.3
pip install pytorch-lightning==2.4.0
pip install deepspeed==0.17.0
pip install setuptools==57.5.0

sudo apt-get update -yqq

# Install dependency for calvin
sudo apt-get -yqq install libegl1 libgl1

# Install EGL mesa
sudo apt-get install -yqq libegl1-mesa libegl1-mesa-dev
sudo apt-get install -yqq mesa-utils libosmesa6-dev llvm
sudo apt-get -yqq install meson
sudo apt-get -yqq build-dep mesa

conda install -c conda-forge gcc=12.1.0 gxx_linux-64 -y

git clone --recurse-submodules https://github.com/mees/calvin.git

CALVIN_ROOT=$(pwd)/calvin
cd ${CALVIN_ROOT}
sed -i '11d' calvin_models/requirements.txt
sed -i '12d' calvin_models/requirements.txt
sed -i '13d' calvin_models/requirements.txt
sed -i '14d' calvin_models/requirements.txt

sh install.sh

# CALVIN spesicifcally requires the following version of numpy
pip install numpy==1.23.0

pip install git+https://github.com/openai/CLIP.git
pip install moviepy==1.0.3
pip install networkx==2.6.3
pip install termcolor
pip uninstall -y torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install open3d
pip install ipdb

# Download dataset
# cd ${CALVIN_ROOT}/dataset
# sh download_data.sh debug
# sh download_data.sh D
# sh download_data.sh ABC
# sh download_data.sh ABCD