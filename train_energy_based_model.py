

"""
train_energy_based_model.py
"""
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt

from models import FCNet, ConvNet
from langevin import sample_langevin
from data import sample_2d_data
import matplotlib.pyplot as pyplt

pyplt.rcParams["figure.figsize"] = (18, 6)


parser = argparse.ArgumentParser()
# parser.add_argument('dataset', choices=('8gaussians', '2spirals', 'checkerboard', 'rings', 'MNIST'))
# parser.add_argument('model', choices=('FCNet', 'ConvNet'))
parser.add_argument('--dataset', type=str, default='arb3')
parser.add_argument('--model', type=str, default='FCNet')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate. default: 1e-3')
parser.add_argument('--stepsize', type=float, default=0.1, help='Langevin dynamics step size. default 0.1')
parser.add_argument('--n_step', type=int, default=100, help='The number of Langevin dynamics steps. default 100')
parser.add_argument('--n_epoch', type=int, default=100, help='The number of training epoches. default 100')
parser.add_argument('--alpha', type=float, default=1., help='Regularizer coefficient. default 100')
args = parser.parse_args()

func_name = ""
if args.dataset == "arb1":
    func_name = "x^2+y^2 >= 10"
elif args.dataset == "arb2":
    func_name = "sin(x+y) >= 0.5"
elif args.dataset == "arb3":
    func_name = "sin(x^2+y^2) >= 0.5"

# load dataset
N_train = 10000
N_val = 1000
N_test = 5000

X_train = sample_2d_data(args.dataset, N_train)
X_val = sample_2d_data(args.dataset, N_val)
X_test = sample_2d_data(args.dataset, N_test) #[5000,2]

train_dl = DataLoader(TensorDataset(X_train), batch_size=32, shuffle=True, num_workers=0)
val_dl = DataLoader(TensorDataset(X_val), batch_size=32, shuffle=True, num_workers=0)
test_dl = DataLoader(TensorDataset(X_test), batch_size=1, shuffle=False, num_workers=0)

# build model
if args.model == 'FCNet':
    model = FCNet(in_dim=2, out_dim=1, l_hidden=(100, 100), activation='relu', out_activation='linear')
elif args.model == 'ConvNet':
    model = ConvNet(in_chan=1, out_chan=1)
model.cpu()

opt = Adam(model.parameters(), lr=args.lr)
    
# train loop
for i_epoch in range(args.n_epoch):
    l_loss = []
    for pos_x, in train_dl:
        
        pos_x = pos_x.cpu() #[32,2]
        
        neg_x = torch.randn_like(pos_x)
        neg_x = sample_langevin(neg_x, model, args.stepsize, args.n_step, intermediate_samples=False)
        
        opt.zero_grad()
        pos_out = model(pos_x)
        neg_out = model(neg_x)
        
        loss = (pos_out - neg_out) + args.alpha * (pos_out ** 2 + neg_out ** 2)
        loss = loss.mean()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        opt.step()
        
        l_loss.append(loss.item())
    print(np.mean(l_loss))


# Test - get energy and density functions
enum=50
x_ = np.linspace(-5,5,num=enum)
y_ = np.linspace(-5,5,num=enum)
x,y = np.meshgrid(x_,y_)
x = torch.tensor(x)
y = torch.tensor(y)
X_test = torch.empty(enum*enum, 2)
for i in range(enum):
    a = torch.stack([x[i], y[i]], dim=1)
    X_test[enum*i:enum*(i+1), :] = a

test_dl = DataLoader(TensorDataset(X_test), batch_size=enum, shuffle=False, num_workers=0)

z = torch.empty(enum, enum)
for i, test_x in enumerate(test_dl):
    test_x = test_x[0].cpu()
    energy = model(test_x)
    energy = torch.reshape(energy, (-1, ))
    z[i,:] = energy #0: left bottom
print(z)
levels = z.detach().numpy() # energy 
top = np.exp(-levels)
partition = sum(top) / enum*enum 
density = top / partition # density


fig, axs = plt.subplots(1, 3)
axs[0].scatter(X_train[:,0], X_train[:,1])
axs[0].set_xlim([-5, 5])
axs[0].set_ylim([-5, 5])
axs[0].set_title(f"Data {func_name}")
im1 = axs[1].contour(x,y,levels,50)
axs[1].set_title("Energy function")
fig.colorbar(im1, ax=axs[1], orientation='vertical')
im2 = axs[2].contour(x,y,density,50)
fig.colorbar(im2, ax=axs[2], orientation='vertical')
axs[2].set_title("Density function")
fig.tight_layout()
fig.savefig(f'{args.dataset}.png')


# plt.ion()
# x,y = np.meshgrid(x_,y_)
# levels = z.detach().numpy()
# c = plt.contour(x,y,levels,50)
# plt.colorbar()
# plt.title("Energy function")
# plt.savefig(f"{args.dataset}_energy_model.png") 
# plt.close()

# # density function
# top = np.exp(-levels)
# partition = sum(top) / enum*enum 
# density = top / partition
# c = plt.contour(x,y,density,50)
# plt.colorbar()
# plt.title("Density function")
# plt.savefig(f"{args.dataset}_density.png") 







    
