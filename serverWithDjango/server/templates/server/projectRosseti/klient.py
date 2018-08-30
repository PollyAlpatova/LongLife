import socket
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import time

i=int(0)
j=int(0)

sock = socket.socket()

sock.connect(('192.168.0.96', 8000))
ar=np.arange(17)
tab=df(index=ar,columns=[0,1])
while i<16:
    if j==2:
        j=0
        i=i+1
    
    data = sock.recv(1024)
    data=data.decode('utf-8')
    tab.iloc[i,j]=data
    j=j+1
    
print(tab)
tab.to_csv('examp.csv', encoding='utf-8')

index=np.arange(50)
columns=[1,2]
regr=pd.DataFrame(index=index, columns=columns)

ff=0
nn=0
while ff<49:
    if nn==2:
        nn=0
        ff=ff+1
    
    data = sock.recv(1024)
    data=data.decode('utf-8')
    regr.iloc[ff,nn]=data
    nn=nn+1
regr.iloc[0][0]=0
regr.iloc[0][1]=0
print(regr)
regr.to_csv('regr.csv',header=['frst','scnd'], index=False, encoding='utf-8')

    
    
    
    
    
    
    
