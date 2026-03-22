from startercode import load_npy, generate_dataset
from pathlib import Path
import torch
import pandas as pd

labels_df = pd.read_csv('training/train_label_mapping.csv')

# define file paths
AD_path = Path('training/AD')
CN_path = Path('training/CN')

filesAD = sorted([f for f in AD_path.iterdir() if f.is_file()])
filesCN = sorted([f for f in CN_path.iterdir() if f.is_file()])

# containers
data_list = [] # Will hold numpy arrays (brain recordings)
labels = [] # Will hold integers (0 = healthy, 1 = AD)
subject_ids = [] # Will hold integers (subject ID numbers)

# Drop non AD/CN subjects
task_df = labels_df[labels_df['label'].isin(['A', 'C'])].copy()
task_df['binary_label'] = task_df['label'].map({'A': 1, 'C': 0})
valid_ids = set(task_df['anonymized_id'].values)

# Load EEG files
for folder, label_value in [(AD_path, 1), (CN_path, 0)]:
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix != '.npy':
            continue

        subject_id = int(path.stem)

        # skip subjects not in our filtered CSV list
        if subject_id not in valid_ids:
            continue

        data = load_npy(path)  # (channels, time)

        data_list.append(data)
        labels.append(label_value)
        subject_ids.append(subject_id)

        duration_min = data.shape[1] / 128 / 60

# generate dataset
X_rbp, X_scc, y, groups = generate_dataset(
    data_list,
    labels,
    subject_ids,
    sfreq=128
)

# save dataset to pt file
torch.save({
    "X_rbp": X_rbp,
    "X_scc": X_scc,
    "y": y.squeeze(),
    "groups": groups
}, "dataset.pt")