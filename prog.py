import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import pdftables_api
import os
import urllib3
import socket as sk





'''c = pdftables_api.Client('n2fflghmddn0')
c.xlsx('2.pdf', '1.xlsx') 
'''
                    
fr=pd.read_excel('1.xlsx')
dq=df(fr)
print(dq)

ax=plt.subplot(111)
ax.plot(dq,"-o",label='$')
a=np.array([6,6,6,6,])
ax.plot(a,"--r")
plt.title('abc')
ax.legend()
plt.show()

