
#%%%%

import pandas as pd 
import numpy as np
import wandb

#%%
def LoadInfo(paramName, metric_list, project_name='DAFS_v2', sweep='pxcw9b3s'):
    api = wandb.Api()
    runs = api.sweep('zangzelin/'+project_name+ '/' +sweep).runs
    summary_list, config_list, name_list = [], [], []
    
    for run in runs: 
        summary_list.append(run.summary._json_dict)
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

    summary = pd.DataFrame(summary_list)
    config = pd.DataFrame(config_list)
    data = pd.concat([summary, config], axis=1)
    return data[paramName+metric_list]

paramName=[
    'data_name',
    'Bernoulli_t',
    'Normal_t',
    'Uniform_t',
    'num_fea_aim',
    'nu',
    'n_rate',
    'metric/#Feature'
]
metric_list=[
    'mbest/ET_ACC_val_mean',
    'mbest/ET_ACC_test_mean',
]

sweep_list = {
    # 'v2_augtype_Uniform_t': '3pvs87x0', # 'pxcw9b3s',
    # 'v2_augtype_Bernoulli_t': 'kx18kkec', #'fkzunyqs',
    # 'v2_augtype_Normal_t': 'gsd9hyzd', # '4jn30q1q',
    'v2_num_feature': 'wwbw96cf', #'jyr2r525'
}

#%%
def Any(wandb_data):
    print('wandb_data.shape', wandb_data.shape)
    # if wandb_data['Bernoulli_t'].max() > 0:
    #     mode = 'Bernoulli_t'
    # elif wandb_data['Normal_t'].max() > 0:
    #     mode = 'Normal_t'
    # elif wandb_data['Uniform_t'].max() > 0:
    #     mode = 'Uniform_t'

    mode = 'num_fea_aim'

    t_list = list(set(wandb_data[mode]))
    # t_list.remove(0.05)
    t_list.sort()
    data_name_list = list(set(wandb_data['data_name']))
    # data_name_list.remove('arcene')
    # data_name_list.remove('EMnistBC')

    data = np.zeros((len(data_name_list), len(t_list)))
    for i, data_name in enumerate(data_name_list):
        for j, ber in enumerate(t_list):
            try:
                wandb_data_0 = wandb_data[wandb_data[mode] == ber]
                data_select = wandb_data_0[wandb_data_0['data_name'] == data_name]
                val_max = np.array(data_select['mbest/ET_ACC_val_mean'].max())
                test_best = data_select[data_select['mbest/ET_ACC_val_mean']==val_max]['mbest/ET_ACC_val_mean'].max()
                data[i,j] = test_best
            except:
                data[i,j] = float(0)
    
    data_show = pd.DataFrame(data, index=data_name_list, columns=t_list).T
    data_show['Average'] = data_show.mean(axis=1)
    data_show = data_show.T
    print(data_show)
    return data_show


#%%
data_show_list = []
for sweep in sweep_list.keys():
    # print(sweep)
    wandb_data = LoadInfo(
        paramName, 
        metric_list, 
        project_name='OTN', 
        sweep=sweep_list[sweep],
        )
    # wandb_data[wandb_data['metric/#Feature']==64]
    data_show = Any(wandb_data)
    data_show_list.append(data_show)


data_sort = [
    'Coil20',
    'Coil100', 
    'Activity', 
    'Mnist', 
    'KMnist', 
    'EMnistBC', 
    'pixraw10P',
    'Prostatege', 
    'arcene', 
    'HCL60K3037D', 
    'Gast10k1457', 
    'MCAD9119', 
    'PBMCD2638', 
    'Average'
]

#%%
data_show = data_show_list[0]
data_show = (data_show.loc[data_sort]*100).round(1)
data_show.style.highlight_max(color='red', axis=1)
#%%
print(data_show.to_latex())