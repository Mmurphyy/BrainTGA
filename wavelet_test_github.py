import pywt
import numpy as np
import torch

def wavelet_transform(signal, wavelet, level):

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate(coeffs)

def apply_wavelet_transform(pyg_graphs, wavelet, level):

    transformed_graphs = []
    
    for data in pyg_graphs:
        x = data.x.numpy()  
        
        transformed_x = np.array([wavelet_transform(signal, wavelet, level) for signal in x])
        
        transformed_x = torch.tensor(transformed_x, dtype=torch.float)
        
        new_data = data.clone()
        new_data.x = transformed_x
        
        transformed_graphs.append(new_data)
    
    return transformed_graphs


transformed_pyg_graphs_level1 = apply_wavelet_transform(pyg_graphs, wavelet='db1', level=2)

transformed_pyg_graphs_level2 = apply_wavelet_transform(pyg_graphs, wavelet='db1', level=1)


import torch

for i in range(len(pyg_graphs)):
    pyg_graphs[i].x1 = transformed_pyg_graphs_level1[i].x
    pyg_graphs[i].x2 = transformed_pyg_graphs_level2[i].x
    

for graph in pyg_graphs:
    for tensor_name in ['x1', 'x2']:
        tensor = getattr(graph, tensor_name) 
        for row_idx in range(tensor.size(0)):  
            row = tensor[row_idx]
            non_zero_elements = row[row != 0] 
            if non_zero_elements.numel() > 0:  
                row_mean = non_zero_elements.mean()  
                row[row == 0] = row_mean  
        setattr(graph, tensor_name, tensor)  
