#pred_analytics.py
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from lstm_pred_analytics import CNN_with_torch_lstm, train
import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split

def create_dataset(file_name):
    min_steps_to_predict = 100
    with open(file_name, 'r') as file:
        data = file.read()
    json_string = '['+data[1:]+']'
    json_data = json.loads(json_string)
    list_data = []
    for obj in json_data:
        temp_data = []
        for key, value in obj.items():
            if isinstance(value, list):
                temp_data.extend(value)
            else:
                temp_data.append(value)
        list_data.append(temp_data)
    number_of_params = len(list_data[0])
    ys = []
    
    for n in range(number_of_params):
        data_string = np.array(list_data)[:,n]
        temp_y = np.zeros(len(data_string))
        median_val = np.median(data_string)
        border = 0.007*median_val #border of variation to predict
        for i,obj in enumerate(data_string):
            if i >= min_steps_to_predict:
                if obj<median_val-border or obj>median_val+border:
                    temp_y[i-min_steps_to_predict] = 1
        ys.append(temp_y[min_steps_to_predict:]) 
    xs = np.array(list_data)[:-min_steps_to_predict,:] 
    return xs,ys
        
def create_pa_model(path_to_json):
    '''
    Input: path to json created by analize.py
    Output: saved weights of LSTM model and loss graphics
    '''
    xs,ys = create_dataset(path_to_json)
    lstm_step = 30
    datasets = []
    for j,yp in enumerate(ys): 
        if j == 3:
            sub_dataset = []
            l = 0
            for y in yp:
                if l+lstm_step<=len(xs):
                    sub_dataset.append([y,xs[l:l+lstm_step,:]])
                    l+=1
                else:
                    break
            datasets.append(sub_dataset)
            break

    hidden_size = 100
    output_dim = 1
    X = []
    Y = []
    for obj in datasets[0]:
        X.append(obj[1])
        Y.append(obj[0])
    X = np.array(X)
    Y = np.array(Y)
    Y = Y.reshape(-1,1)
    look_back = X.shape[2]
    Y = np.array(Y)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    valid_length = int(0.1*len(dataset))
    train_length = len(dataset) - valid_length
    validation_dataset, train_dataset = random_split(dataset, [valid_length, train_length], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataloader = DataLoader(validation_dataset, batch_size=1)
    model = CNN_with_torch_lstm(look_back, hidden_size, output_dim)
    eps_loss = train(model, train_dataloader, val_dataloader, 70, 0.001)

    plt.plot(eps_loss)
    plt.show()

if __name__=='__main__':
    create_pa_model('/Nornikel/dataanswer.json')
