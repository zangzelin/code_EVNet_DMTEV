import torch


def aug_near_mix(index, dataset, k=10, random_t=0.1, device="cuda"):
    r = (
        torch.arange(start=0, end=index.shape[0]) * k
        + torch.randint(low=1, high=k, size=(index.shape[0],))
    ).to(device)
    
    # import pdb; pdb.set_trace()
    random_select_near_index = (
        dataset.neighbors_index[index][:, :k].reshape((-1,))[r].long()
    )
    random_select_near_data2 = dataset.data[random_select_near_index]
    random_rate = torch.rand(size=(index.shape[0], 1)).to(device) * random_t
    return (
        random_rate * random_select_near_data2 + (1 - random_rate) * dataset.data[index]
    )


def aug_near_feautee_change(index, dataset, k=10, t=0.99, device="cuda"):
    r = torch.arange(start=0, end=index.shape[0], device=device) * k + torch.randint(low=1, high=k, size=(index.shape[0],), device=device)
    
    random_select_near_index = (
        dataset.neighbors_index[index][:, :k].reshape((-1,))[r].long()
    )
    random_select_near_data2 = dataset.data[random_select_near_index]
    data_origin = dataset.data[index]
    random_rate = torch.rand(size=(1, data_origin.shape[1]), device=device)
    random_mask = (random_rate > t).reshape(-1).float()
    return random_select_near_data2 * random_mask + data_origin * (1 - random_mask)


def aug_randn(index, dataset, k=10, t=0.01, device="cuda"):
    data_origin = dataset.data[index]
    return (
        data_origin
        + torch.randn(data_origin.shape, device=data_origin.device) * 0.1 * t
    )
