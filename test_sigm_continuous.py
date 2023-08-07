# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:43:43 2023

@author: casel
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rho = .8  # attendibilità
kappa1 = 0  # media del fattore
phi1 = 1  # varianza del fattore
N = 1000  # numerosità campionaria

epsvar1 = ((1 - rho) * phi1) / rho
n_individuals = 1000
loadings = 0.8
nonlinearity = 2

obs_var = 7  # numero di variabili osservate
nl_obs_var = 1  # numero di variabili osservate con relazioni non lineari
k_var = obs_var + nl_obs_var

N = 1000  # numerosità campionaria


def item_function(x):
    return (1 / (1 + np.exp(-x*3.0))*8)-4


np.random.seed(seed=2022)

X = []
NL_X = []
KSI = []
KSI_NL = []

for i in range(N):
    ksi = np.random.normal(0, 1)
    KSI.append(ksi)
    for j in range(obs_var):
        x = 0 + 0.8 * ksi + np.random.normal(0, epsvar1)
        X.append(x)
    for z in range(nl_obs_var):
        nl_x = item_function(ksi) * loadings + np.random.normal(0, epsvar1)

        NL_X.append(nl_x)


X = np.array(X)
NL_X = np.array(NL_X)
KSI = np.array(KSI)

#print(np.shape(KSI))
X = X.reshape(N, obs_var)
NL_X = NL_X.reshape(N, nl_obs_var)

features = 1
inputs = 8
categories = 1
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inputs * categories, 4),
            nn.Tanh(),
            nn.Linear(4, features * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(features, 4),
            nn.Tanh(),
            nn.Linear(4, inputs * categories),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # Encoding

        h = self.encode(x).view(-1, 2, features)
        # get `mu` and `log_var`
        mu = h[:, 0, :]  # the first feature values as mean
        logvar = h[:, 1, :]  # the other feature values as variance
        z = self.reparameterize(mu, logvar)
        # Decoding
        x_hat = self.decode(z)
        return x_hat, mu, logvar



my_data = np.append(X, NL_X, axis=1)



def loss_function(x_hat, x, mu, logvar):
    # Calcola la binary cross entropy tra l'input originale e l'output ricostruito
    mae = nn.MSELoss(reduction="sum")
    MAE = mae(x_hat, x)

    # Calcola la divergenza di Kullback-Leibler tra la distribuzione latente e una distribuzione normale
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Calcola la loss totale come somma dei due termini
    loss = MAE + KLD

    return loss





batch_size = 32
from factor_analyzer import FactorAnalyzer
c = my_data

df_features = pd.DataFrame(c)
fa = FactorAnalyzer(n_factors=1,rotation="varimax")
fa.fit(df_features)
print(fa.loadings_)


tensor_x = torch.Tensor(my_data)  # transform to torch tensor

my_dataset = TensorDataset(tensor_x, tensor_x)  # create your datset

train_loader = DataLoader(my_dataset, batch_size=32, shuffle=False)  # create your dataloader

np.shape(tensor_x)
# Creazione modello e definizione ottimizzatore
model = VAE()
model = torch.load("sigm_continuous.pt").to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)



model.eval()
latent_space = []
with torch.no_grad():
    for batch_idx, data in enumerate(train_loader):
        inputs, _ = data
        inputs = inputs.to(device)
        _, mu, _ = model(inputs)
        latent_space.append(mu.cpu().detach().numpy())

latent_space = np.concatenate(latent_space, axis=0)
plt.figure()
ax = plt.axes()
sns.kdeplot(KSI, ax=ax,color='red',label='true scores')
sns.kdeplot(latent_space, ax=ax,color='blue',label='estimated scores')
plt.xlabel('factor scores', fontsize=12)
plt.ylabel('density', fontsize=12)
plt.legend()
plt.savefig("VAE_scores_one_non_linear")
rec = []
for i in range(np.shape(my_data)[0]):
    data, _ = my_dataset[i:i + 1]
    data = data.to(device)
    data = data.view(data.size(0), -1)
    y, _, _ = model(data)
    rec.append(y[0].cpu().detach().numpy())

rec_train = np.array(rec)

RMSE = mean_squared_error(my_data, rec_train, squared=False)

Expl_Var = explained_variance_score(my_data, rec_train)



corr = np.corrcoef(rec_train, latent_space, rowvar=False)
loadings = corr[0:8, 6:8]

corr_VAE_KSI = np.corrcoef(latent_space, KSI, rowvar=False)
print("corr",corr_VAE_KSI)
corr_DATA_VAE = np.corrcoef(my_data, latent_space, rowvar=False)
plt.figure()
plt.scatter(KSI, my_data[:, 7])
plt.scatter(latent_space, rec_train[:, 7])


# plt.scatter(PCA_reconstructed[:,7], PCA_scores, color = "red", alpha = 0.5)


def fit_function(x,a):
    return ((((1 / (1 + np.exp(-x*3.0)))*8)-4) * a)

def linear_fit_function(x,a):
    return  (a * x)

latent_space = np.squeeze(latent_space)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(KSI,latent_space))
input("ws")
popt_nl, pcov = curve_fit(fit_function, latent_space, rec_train[:, -1])
print(popt_nl)
print("cov",pcov)
loadings_fit_nl = popt_nl[0]
loadings_fit = []
for j in range(obs_var):
    popt, pcov = curve_fit(linear_fit_function, latent_space, rec_train[:, j])
    print(popt)
    print("cov", pcov)
    loadings_fit.append(popt[0])


fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(10,6))
i=0

for row in ax:
    for col in row:
        col.scatter(KSI, my_data[:,i],alpha=0.5)
        if i==7:
            col.scatter(latent_space,rec_train[:,i],color="red",alpha=0.5)#
            col.scatter(latent_space,fit_function(latent_space,loadings_fit_nl),color="black",s=3)
        else:
            col.scatter(latent_space,rec_train[:,i],color="red",alpha=0.5)#
            col.plot(KSI,linear_fit_function(KSI,loadings_fit[i]),color="black")
        if i==0:
            col.set_xlabel("Factor scores")
            col.set_ylabel("Item values")
        i+=1

plt.savefig("VAE_relation_one_non_linear")
plt.show()












