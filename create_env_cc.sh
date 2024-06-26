module load python/3.11

virtualenv --no-download --clear ~/pyenv/musmt

source ~/pyenv/musmt/bin/activate

pip install -r requirements.txt