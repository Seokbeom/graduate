#pip install tensorflow.tensorboard  # install tensorboard
#pip show tensorflow.tensorboard
# Location: c:\users\<name>\appdata\roaming\python\python35\site-packages
# now just run tensorboard as:
#python c:\users\<name>\appdata\roaming\python\python35\site-packages\tensorboard\main.py --logdir=<logidr>
#tensorboard --logdir C:/Users/seok0/Desktop/graduate_project/graduate/s2s_attention_train/logs

import csv
import numpy as np
import matplotlib.pyplot as plt


loc = './csv/'
A_p= 'run_movie_dialogue_15_T_9752_OHE_OHE_ATT_-tag-perplexity'
B_p= 'run_movie_dialogue_15_T_9752_OHE_OHE_NOA_-tag-perplexity'
C_p='run_movie_dialogue_15_T_9752_W2V_ORI_ATT_-tag-perplexity'
C_p='run_movie_dialogue_15_T_9752_W2V_NOM_ATT_-tag-perplexity'
D_p='run_movie_dialogue_15_T_9752_W2V_ORI_NOA_-tag-perplexity'

A= np.loadtxt(open(loc + A_p +'.csv', "rb"), delimiter=",", skiprows=1)[:73].T
B= np.loadtxt(open(loc + B_p +'.csv', "rb"), delimiter=",", skiprows=1)[:73].T
C=  np.loadtxt(open(loc + C_p +'.csv', "rb"), delimiter=",", skiprows=1)[:73].T
D=  np.loadtxt(open(loc + D_p +'.csv', "rb"), delimiter=",", skiprows=1)[:73].T
plt.figure()
plt.plot(A[1],A[2], 'b-',label='A (OHE + Attention)')
plt.plot(B[1],B[2], 'b:',label='B (OHE + Seq2Seq)')
plt.plot( C[1], C[2], 'r-', label = 'C (W2V + Attention')
plt.plot( D[1], D[2], 'r:', label = 'D (W2V + Seq2Seq')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.savefig(loc+'Perplexity.png')
plt.show()

A_p= 'run_movie_dialogue_15_T_9752_OHE_OHE_ATT_-tag-loss'
B_p= 'run_movie_dialogue_15_T_9752_OHE_OHE_NOA_-tag-loss'
C_p='run_movie_dialogue_15_T_9752_W2V_ORI_ATT_-tag-loss'
C_p='run_movie_dialogue_15_T_9752_W2V_NOM_ATT_-tag-loss'
D_p='run_movie_dialogue_15_T_9752_W2V_ORI_NOA_-tag-loss'
A= np.loadtxt(open(loc + A_p +'.csv', "rb"), delimiter=",", skiprows=1)[:73].T
B= np.loadtxt(open(loc + B_p +'.csv', "rb"), delimiter=",", skiprows=1)[:73].T
C=  np.loadtxt(open(loc + C_p +'.csv', "rb"), delimiter=",", skiprows=1)[:73].T
D=  np.loadtxt(open(loc + D_p +'.csv', "rb"), delimiter=",", skiprows=1)[:73].T
plt.figure()

plt.plot(A[1],A[2], 'b-',label='A (OHE + Attention)')
plt.plot(B[1],B[2], 'b:',label='B (OHE + Seq2Seq)')
plt.plot( C[1], C[2], 'r-', label = 'C (W2V + Attention')
plt.plot( D[1], D[2], 'r:', label = 'D (W2V + Seq2Seq')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(loc+'LOSS.png')
plt.show()