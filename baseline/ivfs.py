import torch
import torch.nn as nn

def Score(X, X_r):
    D_x = nn.functional.pdist(X)
    D_xr = nn.functional.pdist(X_r)
    # print(D_x.shap
    
    return -1*torch.norm(D_x-D_xr, p=float('inf'))
    # return 1*torch.norm(D_x-D_xr)

def ivfs_selector(data, tilde_feature, tilde_sample, k):
    
    data = torch.Tensor(data).to('cpu')
    feature_all = data.shape[1]
    sample_all = data.shape[0]
    
    counter = torch.zeros(feature_all).to('cpu')
    sum_score = torch.zeros(feature_all).to('cpu')
    
    for k_c in range(k):
        if k_c % 10 == 0:
            print('current epoch:', k_c)
        randon_feature_index = torch.randperm(feature_all)[:tilde_feature].to('cpu')
        randon_sample_index = torch.randperm(sample_all)[:tilde_sample].to('cpu')
        
        
        X_k = data[randon_sample_index]
        X_kr = data[randon_sample_index][:, randon_feature_index]
        sum_score[randon_feature_index] += Score(X_k, X_kr)
        counter[randon_feature_index] += 1
    
    score = sum_score/counter
    return score.detach().cpu().numpy()

if __name__ == '__main__':
    data = torch.randn(size=(100,10))
    score = ivfs_selector(data, num_fea=5, tilde_feature=5, tilde_sample=20, k=100)     
        