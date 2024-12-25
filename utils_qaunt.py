import torch
def weight_quant_fn(weight,  num_bits, quant_method, num_std = 2):
    if quant_method=="normal_float":
        return quant_nf4_block(weight,num_bits=num_bits)
    elif quant_method == "uniform":
        mean, std = weight.mean(), weight.std()
        clip_val = (mean - num_std * std, mean + num_std * std)
        clip_val = torch.tensor(list(clip_val))
        return quant_uniform(weight,num_bits,clip_val)
    else:
        raise ValueError("")


from scipy.stats import norm
def create_normal_map(offset=0.9677083, symmetric=False, num_bits = 4):
    variations = 2**num_bits
    if symmetric == True:
        v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
        values = []
        for index in range(len(v) - 1):
            values.append(0.5 * v[index] + 0.5 * v[index + 1])
        v = values
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
        v2 = [0]
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
        v = v1 + v2 + v3


    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()
    return values

def quantize_tensor(X, L):
    X_expanded = X.unsqueeze(-1)
    L_reshaped = torch.tensor(L).reshape(1, -1)
    abs_diff = torch.abs(X_expanded - L_reshaped)
    min_index = torch.argmin(abs_diff, dim=-1)
    return L[min_index]

def quant_nf4(weight,num_bits=2):
    max_abs = torch.abs(weight).max()
    weight_divabs = weight/max_abs
    data_format = create_normal_map(num_bits=num_bits)
    weights_divabs = quantize_tensor(weight_divabs, data_format)
    return weights_divabs*max_abs

def quant_nf4_block(weight, block_size=64, num_bits=2):
    def quant_nf4(weight, num_bits=num_bits):
        max_abs = torch.abs(weight).max()
        weight_divabs = weight / max_abs
        data_format = create_normal_map(num_bits=num_bits)
        weights_divabs = quantize_tensor(weight_divabs, data_format)
        return weights_divabs * max_abs
    weight_resize = weight.resize(weight.shape[0]*weight.shape[1]//block_size,block_size)
    quant_block = torch.vmap(quant_nf4, out_dims=0)
    return quant_block(weight_resize).view(weight.shape[0],weight.shape[1])


def quant_uniform(input, num_bits=2, clip_val = None):
    if clip_val!=None:
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
    alpha = (input.max() - input.min()).detach()
    beta = input.min().detach()
    input_normalized = (input - beta) / (alpha + 1e-8) 
    s = (2 ** num_bits - 1)
    quant_input = torch.round(input_normalized * s).div(s)
    output = quant_input * (alpha + 1e-8) + beta  
    return output
