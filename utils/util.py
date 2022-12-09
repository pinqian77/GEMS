import datetime
import torch

start_date = '2005-01-03'
end_date = '2020-11-30'
date_format = '%Y-%m-%d'
start_datetime = datetime.datetime.strptime(start_date, date_format)
end_datetime = datetime.datetime.strptime(end_date, date_format)
number_datetime = (end_datetime - start_datetime).days + 1

def index_to_date(index,timestamp):
    """
    Args:
        index: the number index
        timestamp: a list of times [2005-01-03, 2005-01-04, ...]
    Returns:
    """
    return timestamp[index]


def date_to_index(date_string,timestamp):
    """
    Args:
        date_string: in format of '2005-01-03'
    Returns: the trade days from start_date: '2005-01-03'
    """
    assert date_string in timestamp, '%s is not a trading day' %(date_string)
    return timestamp.index(date_string)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def bhattacharyya_gaussian_distance(mu1, sigma1, mu2, sigma2):
    _sigma1 = sigma1
    _sigma2 = sigma2
    
    sigma = (1 / 2) * (_sigma1 + _sigma2)
    
    tmp = (mu1-mu2) * (1 / sigma)
    T1 = (1 / 8) * torch.matmul(tmp, tmp.T)
    T1 = torch.diag(T1, 0).reshape((mu1.shape[0],-1))
    
    T2 = (1 / 2) * torch.log(torch.prod(sigma,1) / torch.sqrt(torch.prod(_sigma1,1) * torch.prod(_sigma2,1)))
    T2 = T2.reshape((mu1.shape[0],-1))
    
    return T1 + T2