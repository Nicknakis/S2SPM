# Import all the packages
import argparse
import sys
from tqdm import tqdm
import torch

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

#from blobs import *
from sklearn.decomposition import PCA
# import sparse 
# import stats
import math



sys.path.append('./src/')

from S2SPM import S2SPM_
from link_prediction import LP_

parser = argparse.ArgumentParser(description='Skellam Latent Distance Models')

parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs for training (default: 5K)')


parser.add_argument('--cuda', type=eval, 
                      choices=[True, False],  default=True,
                    help='CUDA training')



parser.add_argument('--LP',type=eval, 
                      choices=[True, False], default=True,
                    help='performs link prediction')

parser.add_argument('--pretrained',type=eval, 
                      choices=[True, False], default=False,
                    help='Uses pretrained embeddings for link prediction (default: True)')

parser.add_argument('--D', type=int, default=8, metavar='N',
                    help='dimensionality of the embeddings (default: 8)')

parser.add_argument('--lr', type=float, default=0.05, metavar='N',
                    help='learning rate for the ADAM optimizer, for large values of delta 0.01 is more stable (default: 0.05)')

parser.add_argument('--sample_percentage', type=float, default=0.3, metavar='N',
                    help='Sample size network percentage, it should be equal or less than 1 (default: 0.3)')



parser.add_argument('--dataset', type=str, default='Sapiens',
                    help='dataset to apply Skellam Latent Distance Modeling on')



args = parser.parse_args()

if  args.cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')





plt.style.use('ggplot')


torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    latent_dim=args.D
    dataset=args.dataset
   
    
   
    losses=[]
    data=np.loadtxt("./datasets/undirected/"+dataset+'/edges.txt')
    data[:,0:2].sort(1)
    mask=data[:,0]<data[:,1]
    data=data[mask]
    
    sparse_i=torch.from_numpy(data[:,0]).long().to(device)
    # input data, link column positions with i<j
    sparse_j=torch.from_numpy(data[:,1]).long().to(device)
 

    weights_signed=torch.from_numpy(data[:,2]).long().to(device)
    
   
    
    # network size
    N=int(sparse_j.max()+1)
    print(N)
    # sample size of blocks-> sample_size*(sample_size-1)/2 pairs
    
    sample_size=int(args.sample_percentage*N)
    model = S2SPM_(sparse_i,sparse_j,weights_signed,N,latent_dim=latent_dim,sample_size=sample_size,device=device).to(device)         
    
    # create initial convex hull
    model.find_convex_hull()
    # initialize SLIM-RAA model
    model.LDM_to_RAA()
    
    
    # create initial convex hull
    model.find_convex_hull_w()
    # initialize SLIM-RAA model
    model.LDM_to_RAA_w()
    
    # set-up optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)  

    
    for epoch in tqdm(range(args.epochs),desc="Model is Running…",ascii=False, ncols=75):                 
        
                            
        
        loss=-model.LSM_likelihood_bias_sample(epoch=epoch)/model.sample_size

            
        losses.append(loss.item())
        
        
        
        optimizer.zero_grad() # clear the gradients.   
        loss.backward() # backpropagate
        optimizer.step() # update the weights

        if epoch%500==0:

            pred=LP_(model.latent_z,model.latent_w,model.gamma,model.delta,dataset,sparse_i,sparse_j,weights_signed,device=device)
            # p@n
            p_n_roc,p_n_pr=pred.pos_neg()
            # p@z
            p_z_roc,p_z_pr=pred.pos_zer()
            # n@z
            n_z_roc,n_z_pr=pred.neg_zer()
            
    if args.LP:
        pred=LP_(model.latent_z,model.latent_w,model.gamma,model.delta,dataset,sparse_i,sparse_j,weights_signed,device=device)
        # p@n
        p_n_roc,p_n_pr=pred.pos_neg()
        # p@z
        p_z_roc,p_z_pr=pred.pos_zer()
        # n@z
        n_z_roc,n_z_pr=pred.neg_zer()

            

 
 


            
            
