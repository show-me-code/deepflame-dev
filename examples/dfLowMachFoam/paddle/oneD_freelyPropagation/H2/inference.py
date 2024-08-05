import paddle
import paddle.nn as nn
import numpy as np
import time
import os
import cantera as ct

device_main = "gpu:0"  
device_list = range(paddle.device.cuda.device_count())
paddle.set_printoptions(precision=10)

class NN_MLP(nn.Layer):
    def __init__(self, layer_info):
        super(NN_MLP, self).__init__()
        self.net = paddle.nn.Sequential()
        n = len(layer_info) - 1
        for i in range(n - 1):
            self.net.add_sublayer('linear_layer_%d' % (i), nn.Linear(layer_info[i], layer_info[i + 1]))
            self.net.add_sublayer('gelu_layer_%d' % (i), nn.GELU())
        self.net.add_sublayer('linear_layer_%d' % (n - 1), nn.Linear(layer_info[n - 1], layer_info[n]))

    def forward(self, x):
        return self.net(x)

try:
    # load variables from constant/CanteraTorchProperties
    path_r = r"./constant/CanteraTorchProperties"
    with open(path_r, "r") as f:
        data = f.read()
        i = data.index('torchModel')
        a = data.index('"', i)
        b = data.index('"', a + 1)
        modelName = data[a + 1: b]

        i = data.index('frozenTemperature')
        a = data.index(';', i)
        b = data.rfind(' ', i + 1, a)
        frozenTemperature = float(data[b + 1: a])

        i = data.index('inferenceDeltaTime')
        a = data.index(';', i)
        b = data.rfind(' ', i + 1, a)
        delta_t = float(data[b + 1: a])

        i = data.index('CanteraMechanismFile')
        a = data.index('"', i)
        b = data.index('"', a + 1)
        mechanismName = data[a + 1: b]

        i = data.index('GPU')
        a = data.index(';', i)
        b = data.rfind(' ', i + 1, a)
        switch_GPU = data[b + 1: a]

    # read mechanism species number
    gas = ct.Solution(mechanismName)
    n_species = gas.n_species

    # load OpenFOAM switch
    switch_on = ["true", "True", "on", "yes", "y", "t", "any"]
    switch_off = ["false", "False", "off", "no", "n", "f", "none"]
    if switch_GPU in switch_on:
        paddle.device.set_device(device_main)
        device_ids = device_list
        # paddle.set_device(device_main)
    elif switch_GPU in switch_off:
        paddle.device.set_device("cpu")
        device_ids = [0]
    else:
        print("invalid setting!")
        os._exit(0)

    lamda = 0.1
    dim = 9

    state_dict = paddle.load(modelName)
    Xmu0 = state_dict['data_in_mean']
    Xstd0 = state_dict['data_in_std']
    Ymu0 = state_dict['data_target_mean']
    Ystd0 = state_dict['data_target_std']

    Xmu0 = paddle.to_tensor(Xmu0).unsqueeze(0).astype('float32')
    Xstd0 = paddle.to_tensor(Xstd0).unsqueeze(0).astype('float32')
    Ymu0 = paddle.to_tensor(Ymu0).unsqueeze(0).astype('float32')
    Ystd0 = paddle.to_tensor(Ystd0).unsqueeze(0).astype('float32')

    Xmu1 = Xmu0
    Xstd1 = Xstd0
    Ymu1 = Ymu0
    Ystd1 = Ystd0

    Xmu2 = Xmu0
    Xstd2 = Xstd0
    Ymu2 = Ymu0
    Ystd2 = Ystd0

    # load model
    layers = [n_species + 2, 1600, 800, 400, 1]

    model0list = []
    for i in range(n_species - 1):
        model0list.append(NN_MLP(layers))

    for i in range(n_species - 1):
        model0list[i].set_state_dict(state_dict[f'net{i}'])

    for i in range(n_species - 1):
        model0list[i].eval()
        model0list[i] = paddle.DataParallel(model0list[i]) if len(device_list) > 1 else model0list[i]

except Exception as e:
    print(e.args)


def inference(vec0):
    '''
    use model to inference
    '''
    vec0 = np.abs(np.reshape(vec0, (-1, 3 + n_species)))  # T, P, Yi(7), Rho
    vec0[:, 1] *= 101325
    mask = vec0[:, 0] > frozenTemperature
    vec0_input = vec0[mask, :]
    print(f'real inference points number: {vec0_input.shape[0]}')

    try:
        with paddle.no_grad():
            input0_ = paddle.to_tensor(vec0_input).astype('float32')

            # pre_processing
            rho0 = input0_[:, -1].unsqueeze(1)
            input0_Y = input0_[:, 2:-1].clone()
            input0_bct = input0_[:, 0:-1]
            input0_bct[:, 2:] = (input0_bct[:, 2:]**(lamda) - 1) / lamda  # BCT
            input0_normalized = (input0_bct - Xmu0) / Xstd0
            input0_normalized = input0_normalized.astype('float32')

            output0_normalized = []
            for i in range(n_species - 1):
                output0_normalized.append(model0list[i](input0_normalized))
            output0_normalized = paddle.concat(output0_normalized, axis=1)

            output0_bct = output0_normalized * Ystd0 + Ymu0 + input0_bct[:, 2:-1]
            output0_Y = input0_Y.clone()
            output0_Y[:, :-1] = (lamda * output0_bct + 1)**(1 / lamda)
            output0_Y[:, :-1] = output0_Y[:, :-1] / paddle.sum(output0_Y[:, :-1], axis=1, keepdim=True) * (1 - output0_Y[:, -1:])
            output0 = (output0_Y - input0_Y) * rho0 / delta_t
            output0 = output0.numpy()

            result = np.zeros((vec0.shape[0], n_species))
            result[mask, :] = output0
            return result
    except Exception as e:
        print(e.args)
