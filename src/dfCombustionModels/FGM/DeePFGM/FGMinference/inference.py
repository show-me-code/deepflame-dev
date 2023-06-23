import numpy as np
import torch
from scipy import stats
from scipy.special import inv_boxcox
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class BasicBlock(nn.Module):
    def __init__(self, n_input,n_hidden):
        super(BasicBlock, self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_input)
        self.relu = nn.ReLU()
 
    def forward(self, x):
        identity = x
 
        out = self.hidden1(x)
        out = self.relu(out)
        out = self.hidden2(out)
        out += identity
        out = self.relu(out)
 
        return out
class ResNet(nn.Module):
    def __init__(self,block,block_num,neuron_num,n_input,n_output):
        super(ResNet, self).__init__()
        self.neuron=neuron_num
        self.input_layer=nn.Linear(n_input,self.neuron)
        self.res_layer = self._make_layer(block,block_num)
        self.output_layer=nn.Linear(self.neuron,n_output)
    def _make_layer(self, block,block_num):
        downsample = None
        layers = []
        for i in range(1, block_num):
            layers.append(block(self.neuron,self.neuron)) 
        return nn.Sequential(*layers)
    def forward(self,input):
        out = self.input_layer(input)
        out = F.relu(out)
        out = self.res_layer(out)
        out = F.relu(out)
        out = self.output_layer(out)
        return out   
def FGM(z,c,gz,gc,gzc,phinum,dimension):
    try:
        z=np.array(z)
        c=np.array(c)
        gz=np.array(gz)
        gc=np.array(gc)
        gzc=np.array(gzc)
        
        # layers_s=[6,6,6,6,6,6,6,6,6,6,6]
        # neurons_s=[200,200,200,100,100,100,100,100,100,200,100]
        name_s=["omegac","omegac","omegac","cp","cp","cp","T","T","YH2O","YCO","YCO2"]
        phis_s=[3,3,3,3,3,3,2,2,1,1,1]
        diff_s=[0,0,0,3,3,3,6,6,8,9,10]
        name=name_s[phinum]
        layers=9
        neurons=300
        phis=phis_s[phinum]
        diff=diff_s[phinum]
        dimension=4
        worktype="data-4D"
        networktype="networks-4D"
        x=np.stack((z, c , gz ,gc), axis=1)
        xmax=np.load("/home/whx/deepflame-dev/src/dfCombustionModels/FGM/DeePFGM/FGMinference/"+worktype+"/process_params/xmax.npy")
        xmin=np.load("/home/whx/deepflame-dev/src/dfCombustionModels/FGM/DeePFGM/FGMinference/"+worktype+"/process_params/xmin.npy")
        for i in range(4):
            x[:,i]=(x[:,i]-xmin[i])/(xmax[i]-xmin[i])
        x = torch.tensor(x,dtype=torch.float)
        # print(np.shape(x))
        # print(name)
        #     device = torch.device("cuda")          # 使用CUDA
        # else:
        # device = torch.device("cpu")           # 使用CPU
        # print("当前设备：",device)
        x=x.to("cuda")
        model = ResNet(BasicBlock,layers,neurons,dimension,phis)
        params = torch.load("/home/whx/deepflame-dev/src/dfCombustionModels/FGM/DeePFGM/FGMinference/"+networktype+"/model_res_"+name+".pth") # 加载参数
        model.load_state_dict(params) # 应用到网络结构中
        model=model.to("cuda")
        # print(params)
        # print("begin inference")
        with torch.no_grad():   
            predictions=model(x)
        # print("end inference")
        predictions=predictions.to("cpu")
        predictions=predictions.data.numpy()
        phimax=np.load("/home/whx/deepflame-dev/src/dfCombustionModels/FGM/DeePFGM/FGMinference/"+worktype+"/process_params/phimax_"+name+".npy")
        phimin=np.load("/home/whx/deepflame-dev/src/dfCombustionModels/FGM/DeePFGM/FGMinference/"+worktype+"/process_params/phimin_"+name+".npy")
        lambdas=np.load("/home/whx/deepflame-dev/src/dfCombustionModels/FGM/DeePFGM/FGMinference/"+worktype+"/process_params/lambdas_"+name+".npy")
        constants=np.load("/home/whx/deepflame-dev/src/dfCombustionModels/FGM/DeePFGM/FGMinference/"+worktype+"/process_params/constants_"+name+".npy")
        ind=phinum
        predictions[:,ind-diff]=predictions[:,ind-diff]*(phimax[ind]-phimin[ind])+phimin[ind]
        # print("begin inv_coxbox")
        predictions[:,ind-diff]=inv_boxcox(predictions[:,ind-diff], lambdas[ind])-constants[ind]
        # print("end inv_coxbox")
        # phi_mean=np.mean(ytrue[:,ind])
        nan_positions = np.isnan(predictions[:,ind-diff])
        # non_nan_positions = ~nan_positions
        # 从原始数组中提取所有非NaN元素
        predictions[nan_positions,ind-diff]=0
        result=predictions[:,ind-diff]
        result = result.tolist()
        # if(phinum==0):
        #     print("x",x)
        #     print("z",z)
        #     print("c",c)
            # print("res",result)
        return result
        
    except Exception as e:
        print(e.args)