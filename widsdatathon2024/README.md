# Install Dependencies
In the root directory of the project, run 
```
python3 -m venv venv2024
source venv2024/bin/activate
python3 -m pip install -r widsdatathon2024/requirements.txt
```

# Download Data
Before using the Kaggle CLI, make sure
1. your API token can be found at the right place, e.g., `~/.kaggle/kaggle.json` on Linux
2. the shebang line in the kaggle script (`venv2024/bin/kaggle`) is `#!/usr/bin/env python3` instead of a hardcoded path

In the root directory of the project, run 
```
kaggle competitions download -c widsdatathon2024-challenge1
mkdir widsdatathon2024/data
unzip widsdatathon2024-challenge1.zip -d widsdatathon2024/data
```