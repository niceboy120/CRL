import os
import re
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

for dataset_name in [name for name in os.listdir() if os.path.isdir(name)]:
    cur_dir = os.path.join(dataset_name, 'ctrl/logs')
    files = {f:open(os.path.join(cur_dir, f)) for f in sorted(os.listdir(cur_dir)) if '.log' in f}
    all_data = {}
    for file_name, file in files.items():
        cur_data = {'file_name':file_name, 'diversity':[], 'rating':[]}
        aug_rate = 0
        for line in file.readlines():
            # match augment rate
            matched = re.findall(r'"augment_rate": \d+', line)
            if len(matched) > 0:
                aug_rate = int(matched[0].split(':')[-1])
            # match logged data
            matched = re.findall(r'\{.*\}', line)
            if len(matched) > 0:
                log_data = json.loads(matched[0].replace("'", '"'))
                for key in ['diversity', 'rating']:
                    cur_data[key].append(log_data[key])
        all_data[f'augment={aug_rate}'] = cur_data

    plt.xlabel('diversity')
    plt.ylabel('rating')
    for key, data in all_data.items():
        plt.scatter(data['diversity'], data['rating'], label=key, alpha=np.linspace(0.1, 1, len(data['rating'])))
    leg = plt.legend()
    for lh in leg.legend_handles: 
        lh._alpha = 1
        # lh.pchanged()
    plt.savefig(os.path.join(cur_dir, f'{dataset_name}-plot.pdf'), format='pdf')
    plt.close()