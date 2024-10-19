## Introduction on kaggle
[WiDS Datathon 2024 Challenge #1](https://www.kaggle.com/competitions/widsdatathon2024-challenge1/overview)

## Install & Run kaggle APIs
In the root directory of the project,
```
# install
python3 -m venv Wids2024DevEnv
source Wids2024DevEnv/bin/activate
python3 -m pip install kaggle
```
```
# make sure token is placed at ~/.kaggle/kaggle.json
# change the shebang line in the kaggle script from a hardcoded path to #!/usr/bin/env python3
```
```
# download data
kaggle competitions download -c widsdatathon2024-challenge1
mkdir data
unzip widsdatathon2024-challenge1.zip -d data
```
