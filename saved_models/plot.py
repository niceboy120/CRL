import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset
import seaborn as sns
sns.set_style('whitegrid')

plot_pareto = False

for dataset_name in [name for name in os.listdir() if os.path.isdir(name)]:
    cur_dir = os.path.join(dataset_name, 'ctrl/logs')
    files = {f:open(os.path.join(cur_dir, f)) for f in os.listdir(cur_dir) if '.log' in f}
    all_data, order = {}, []
    dataset_pareto = {'diversity':[], 'rating':[]}
    for file_name, file in files.items():
        cur_data = {'file_name':file_name, 'diversity':[], 'rating':[]}
        aug_type, aug_rate, aug_strategies = 'mat', 0, ['random']
        file_content = file.read()

        # match dataset pareto
        if len(dataset_pareto['diversity']) == 0:
            matched = re.findall(r'All Pareto Fronts for Env are: \[\[[\s\S]*?\]\]', file_content)
            if len(matched) > 0:
                matched = matched[0][len('All Pareto Fronts for Env are: ['):-1]
                matched = matched.replace('[',' ').replace(']',' ')
                matched = matched.split()
                assert len(matched) % 2 == 0
                for idx in range(0, len(matched), 2):
                    dataset_pareto['rating'].append(float(matched[idx]))
                    dataset_pareto['diversity'].append(float(matched[idx + 1]))
                print('Matched dataset Pareto front:', dataset_pareto)

        # match augment rate
        matched = re.search(r'"augment_rate": \d+(.\d+)?', file_content)
        if matched:
            aug_rate = float(matched.group().split(':')[-1])
            
        # match augment strategies
        matched = re.findall(r'"augment_strategies": \[[\s\S]*?\]', file_content)
        if len(matched) > 0:
            aug_strategies = json.loads('{' + matched[0] + '}')['augment_strategies']
        
        for line in file_content.split('\n'):
            # match aug type:
            if 'augment_type' in line:
                if 'mat' in line:
                    aug_type = 'mat'
                elif 'seq' in line:
                    aug_type = 'seq'
                else:
                    print('Warning: unknwon augment type:', line)
            # match logged data
            matched = re.findall(r'\{.*\}', line)
            if len(matched) > 0:
                log_data = json.loads(matched[0].replace("'", '"'))
                for key in ['diversity', 'rating']:
                    cur_data[key].append(log_data[key])
        if len(cur_data['diversity']) == 0:
            continue

        # donot show old augment data
        if aug_rate >= 1 and len(aug_strategies) > 1:
            continue
        all_data[f'aug:{aug_type}={aug_rate}_{"_".join(aug_strategies)}'] = cur_data
        order.append((aug_rate, len(aug_strategies), aug_type))

    plt.xlabel('diversity')
    plt.ylabel('rating')
    sorted_index = [i[0] for i in sorted(enumerate(order), key=lambda x:x[1])]
    sorted_keys = [list(all_data.keys())[idx] for idx in sorted_index]
    all_data = {key:all_data[key] for key in sorted_keys}
    fig, axs = plt.subplots(2, figsize=(4,8))
    if plot_pareto:
        # plot pareto front
        axs[0].plot(dataset_pareto['diversity'], dataset_pareto['rating'], '*--', label='Pareto', color='darkslategrey')
        axs[1].scatter(dataset_pareto['diversity'], dataset_pareto['rating'], marker='*', label='Pareto', color='darkslategrey')
    else:
        print("NOTE: Pareto front will not be plot, pls refer to the code")
    for key, data in all_data.items():
        # plot pareto front
        data_df = pd.DataFrame({i:data[i] for i in ['diversity', 'rating']})
        mask = paretoset(data_df, sense=["max", "max"])
        front_data = data_df[mask].sort_values(by='diversity')
        axs[0].plot(front_data['diversity'], front_data['rating'], 'o--', label=key)
        axs[1].scatter(data['diversity'], data['rating'], label=key, alpha=np.linspace(0.1, 1, len(data['rating'])))
    axs[0].set_title('Pareto Front')
    axs[1].set_title('Scatter plot (opaque indicates iter)')
    axs[0].legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    leg = axs[1].legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    for lh in leg.legend_handles: 
        lh._alpha = 1
        # lh.pchanged()
    plt.savefig(os.path.join(cur_dir, f'{dataset_name}-plot.pdf'), format='pdf', bbox_inches="tight")
    plt.close()