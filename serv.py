import socket as sk
import time
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn import linear_model
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import sklearn
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import linregress

name=str(input())
named=str(input())
df = pd.read_excel(name+'.xlsx')
dq = pd.read_excel(name+'T.xlsx')
dar = pd.read_excel(named+'.xlsx')
daa = pd.read_excel(named+'T.xlsx')
print(df)
print(dq)
print(dar)
print(daa)
model = linear_model.LinearRegression()

i=0
j=0
CH=np.zeros(2000)
R=np.zeros(2000)
while j<1999:
    ch=df.iat[j,0]
    CH[j]=ch
    j=j+1
CH=CH.reshape(1, -1)
CH=CH[0]

print (CH)
while i<1999:
    r=df.iat[i,1]
    R[i]=r
    i=i+1
R=R.reshape(1, -1)
R=R[0]

print (R)

f=0
d=0
CHT=np.zeros(2000)
RT=np.zeros(2000)
while f<1999:
    cht=dq.iat[f,0]
    CHT[f]=cht
    f=f+1
CHT=CHT.reshape(1, -1)
CHT=CHT[0]

print (CHT)
while d<1999:
    rt=dq.iat[d,1]
    RT[d]=rt
    d=d+1
RT=RT.reshape(1, -1)
RT=RT[0]

print (RT)

fa=0
da=0
CHR=np.zeros(2000)
RR=np.zeros(2000)
while fa<1999:
    chrr=dar.iat[fa,0]
    CHR[fa]=chrr
    fa=fa+1
CHR=CHR.reshape(1, -1)
CHR=CHR[0]

print (CHR)
while da<1999:
    rr=dar.iat[da,1]
    RR[da]=rr
    da=da+1
RR=RR.reshape(1, -1)
RR=RR[0]
print (RR)

fr=0
dr=0
CHRT=np.zeros(2000)
RRT=np.zeros(2000)
while fr<1999:
    chrt=daa.iat[fr,0]
    CHRT[fr]=chrt
    fr=fr+1
CHRT=CHRT.reshape(1, -1)
CHRT=CHRT[0]

print (CHRT)
while dr<1999:
    rrt=daa.iat[dr,1]
    RRT[dr]=rrt
    dr=dr+1
RRT=RRT.reshape(1, -1)
RRT=RRT[0]
print (RRT)

mod1=model.fit(CHRT.reshape(1, -1),RRT.reshape(1, -1))

'''
CHtrn= train_test_split(CH, test_size=0.4)

Rtrn= train_test_split(R, test_size=0.4)
'''

mod=model.fit(CHT.reshape(1, -1),RT.reshape(1, -1))

print(sklearn.metrics.mean_absolute_error(CH,mod.predict(R.reshape(1,-1))[0]))
print('Коэффициенты: \n', mod.coef_)



ax=plt.subplot(111)
#ax.scatter(CH,R,  color='black')
#ax.plot(CH.reshape(1, -1)[0],mod.predict(CH.reshape(1,-1))[0], color='blue',linewidth=2)
dataset=mod.predict(mod1.predict(RR.reshape(1,-1)))[0]
series=mod.predict(mod1.predict(RR.reshape(1,-1)))[0]

def exponential_smoothing(series,alpha):
    result = [series[0]] 
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
for alpha in [0.3, 0.05]:
        ax.plot(exponential_smoothing(dataset, alpha), label="Alpha {}".format(alpha))
        ax.plot([0,2000],[25,25],'--',color='red')

plt.xticks(())
plt.yticks(())

a=np.arange(2000).reshape(1,-1)
a[0][1]=2072
a=mod.predict(mod1.predict(a))

b=np.arange(2000).reshape(1,-1)
b[0][1]=21
b=mod.predict(b)

f=linregress(R,RR)

print(f)
print('b:',b[0][1])
print('a:',a[0][1])
regr=pd.DataFrame()
regr[0]=np.arange(2000)
regr[1]=exponential_smoothing(dataset, alpha)

print(regr)
#Це сервер

sock = sk.socket()
sock.bind(('', 8000))
while True:
    sock.listen(10)
    conn, addr = sock.accept()
    print ('connected:', addr)
    
    fr=pd.read_excel('pas.xlsx')
    ds=pd.DataFrame(fr)
    print (ds)
    ii=0
    jj=0
    while ii<16:
        if jj==2:
            jj=0
            ii=ii+1
        data=ds.iat[ii,jj]
        data=str(data)
        data=data.encode('utf-8')
        conn.send(data.upper())
        jj=jj+1
        time.sleep(0.3)
    newTab='nt'
    newTab=newTab.encode('utf-8')
    conn.send(newTab.upper())
    ff=0
    nn=0
    while nn<49:
        if ff==2:
            ff=0
            nn=nn+1
        data=regr.iat[nn,ff]
        data=str(data)
        data=data.encode('utf-8')
        conn.send(data.upper())
        ff=ff+1
        time.sleep(0.3)
    
    
    


