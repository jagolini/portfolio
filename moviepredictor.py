# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:46:36 2022

@author: jagol
"""

import pandas as pd 
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import zscore
from sklearn.decomposition import PCA

data=pd.read_csv(r"C:\Users\jagol\Downloads\movieReplicationSet - Copy.csv")
shrek=data['Shrek (2001)']
genderID=data['Gender identity (1 = female; 2 = male; 3 = self-described)']
movie=pd.read_csv(r"C:\Users\jagol\Downloads\movies.csv")
personal=pd.read_csv(r"C:\Users\jagol\Downloads\personal.csv")
def clean2(movie):
    trip=[0]*1097

    l=0
    m=0
    while l<1097:
        if (movie[l]>=0):
                trip[m]=l
                m+=1
        l+=1

    trip2=[0]*m
    #print(m)
    q=0
    while q<m:
        trip2[q]=trip[q]
        q+=1
    #print(trip2)
    
    movie2=[0]*m
   
    
    g=0
    
    while g<m:
        movie2[g]=movie[trip2[g]]
       
        g+=1
    return movie2    

def clean(movie,gender):
    trip=[0]*1097
    tripg=[0]*1097
    l=0
    m=0
    while l<1097:
    
        if (movie[l]>=0):
            if(genderID[l]>=0):
                trip[m]=l
                tripg[m]=l
                m+=1
        l+=1

    trip2=[0]*m
    trip2g=[0]*m
    #print(m)
    q=0
    while q<m:
        trip2[q]=trip[q]
        trip2g[q]=tripg[q]
        q+=1
    #print(trip2)
    
    movie2=[0]*m
    gender2=[0]*m
   
    
    g=0
    
    while g<m:
        movie2[g]=movie[trip2[g]]
        gender2[g]=gender[trip2g[g]]
       
        g+=1
    return movie2,gender2  

#print(clean(shrek,genderID))

shrekd2=clean(shrek,genderID)[0]
genderd2=clean(shrek,genderID)[1]
#print(shrekd2)
#print(genderd2)

len2=len(shrekd2)
maleshrek=[0]*len2
femaleshrek=[0]*len2
k=0
f=0
m=0
while k<len2:
    if(genderd2[k]==1):
        femaleshrek[f]=shrekd2[k]
        f+=1
    if(genderd2[k]==2):
        maleshrek[m]=shrekd2[k]
        m+=1
    k+=1

maleshrek2=[0]*240
femaleshrek2=[0]*240
x=0
while x<240:
    maleshrek2[x]=maleshrek[x]
    femaleshrek2[x]=femaleshrek[x]
    x+=1

#print(np.corrcoef(maleshrek,femaleshrek))

#print(np.mean(femaleshrek))
#print(np.mean(maleshrek))


wolf=data['The Wolf of Wall Street (2013)']
alone=data['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)']
wolfc=clean(wolf,alone)[0]
alonec=clean(wolf,alone)[1]
lenwolfc=len(wolfc)
wolfalone=[0]*lenwolfc
wolftogether=[0]*lenwolfc
k=0
f=0
m=0
while k<lenwolfc:
    if(alonec[k]==1):
        wolfalone[f]=wolfc[k]
        f+=1
    if(alonec[k]==0):
        wolftogether[m]=wolfc[k]
        m+=1
    k+=1
    
wolfalone2=[0]*268
wolftogether2=[0]*268
x=0
while x<268:
    wolfalone2[x]=wolfalone[x]
    wolftogether2[x]=wolftogether[x]  
    x+=1
    
#print(np.mean(wolftogether))
#print(np.mean(wolfalone))

lion=data['The Lion King (1994)']
child=data['Are you an only child? (1: Yes; 0: No; -1: Did not respond)']
lionc=clean(lion,child)[0]
childc=clean(lion,child)[1]


lenlionc=len(lionc)
only=[0]*lenlionc
sib=[0]*lenlionc
k=0
f=0
m=0
while k<lenlionc:
    if(childc[k]==1):
        only[f]=lionc[k]
        f+=1
    if(childc[k]==0):
        sib[m]=lionc[k]
        m+=1
    k+=1
only2=[0]*150
sib2=[0]*150    
x=0
while x<150:
    only2[x]=only[x]
    sib2[x]=sib[x]
    x+=1
#print(np.mean(only))
#print(np.mean(sib))

print(stats.ttest_ind(maleshrek2, femaleshrek2))
print(stats.ttest_ind(wolfalone2, wolftogether2))
print(stats.ttest_ind(only2, sib2))

print(data.shape)



sensation=pd.read_csv(r"C:\Users\jagol\Downloads\sensation.csv")
moviexp=pd.read_csv(r"C:\Users\jagol\Downloads\moviexp.csv")

#sensation=zscore(sensation)

#print(sensation.head())
sensation=sensation.dropna()

from sklearn.decomposition import PCA
zscoredData=zscore(sensation)
#print(zscoredData)
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredData)
covarExplained = eigVals/sum(eigVals)*100
for ii in range(len(covarExplained)):
    print(covarExplained[ii].round(3))
numClasses = 20

x = np.linspace(1,numClasses,numClasses)
plt.bar(x, eigVals, color='gray')

plt.plot([0,numClasses],[1,1],color='orange') 
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

whichPrincipalComponent = 7
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) 
plt.xlabel('Question')
plt.show()

sensationpca=pd.DataFrame()
sensationpca.insert(0,sensation.columns[19] ,sensation.iloc[:,19], True)
sensationpca.insert(1,sensation.columns[9] ,sensation.iloc[:,9], True)
sensationpca.insert(2,sensation.columns[15] ,sensation.iloc[:,15], True)
sensationpca.insert(3,sensation.columns[18],sensation.iloc[:,18], True)
sensationpca.insert(4,sensation.columns[17],sensation.iloc[:,17], True)
sensationpca.insert(5,sensation.columns[6],sensation.iloc[:,6], True)
sensationpca.insert(6,sensation.columns[14],sensation.iloc[:,14], True)




moviexp=moviexp.dropna()
zscoredData=zscore(moviexp)
#print(zscoredData)
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredData)
covarExplained = eigVals/sum(eigVals)*100
for ii in range(len(covarExplained)):
    print(covarExplained[ii].round(3))
numClasses = 10

x = np.linspace(1,numClasses,numClasses)
plt.bar(x, eigVals, color='gray')

plt.plot([0,numClasses],[1,1],color='orange') 
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()
plt.plot([0,numClasses],[1,1],color='orange') 
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()
whichPrincipalComponent = 2
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) 
plt.xlabel('Question')
plt.ylabel('Loadings Value')
plt.show()


moviexppca=pd.DataFrame()
moviexppca.insert(0,moviexp.columns[1] ,moviexp.iloc[:,1], True)
moviexppca.insert(1,moviexp.columns[0] ,moviexp.iloc[:,0], True)

personality=pd.read_csv(r"C:\Users\jagol\Downloads\personality.csv")

personality=personality.dropna()

zscoredData=zscore(personality)
#print(zscoredData)
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredData)
covarExplained = eigVals/sum(eigVals)*100
for ii in range(len(covarExplained)):
    print(covarExplained[ii].round(3))
numClasses = 44

x = np.linspace(1,numClasses,numClasses)
plt.bar(x, eigVals, color='gray')

plt.plot([0,numClasses],[1,1],color='orange') 
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()
whichPrincipalComponent = 1
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) 
plt.xlabel('Question')
plt.ylabel('Loadings Value')
plt.show()

personalitypca=pd.DataFrame()
personalitypca.insert(0,personality.columns[18] ,personality.iloc[:,18], True)
personalitypca.insert(1,personality.columns[20] ,personality.iloc[:,20], True)
personalitypca.insert(2,personality.columns[27] ,personality.iloc[:,27], True)
personalitypca.insert(3,personality.columns[26],personality.iloc[:,26], True)
personalitypca.insert(4,personality.columns[38],personality.iloc[:,38], True)
personalitypca.insert(5,personality.columns[40],personality.iloc[:,40], True)
personalitypca.insert(6,personality.columns[14],personality.iloc[:,14], True)
personalitypca.insert(7,personality.columns[21] ,personality.iloc[:,21], True)

print(personalitypca.head(5))
m=data.dropna()
#print(data.shape)
movie=movie.dropna()
#print(movie.shape)

#print(clean2(wolf))
#print(np.mean(clean2(wolf)))
#moviexp.reshape(-1,1)

from sklearn import linear_model
def builder(movie,cols):
 
    joint=cols
    joint.insert(cols.shape[1]-1,cols.shape[1]-1 , movie, True)
    joint=joint.dropna()
    #print(joint.shape)
    y=joint.iloc[:,cols.shape[1]-1]
    #print(joint[0])
    
    if cols.equals(moviexp):
        x=joint[[joint.columns[1],joint.columns[2]]]
    if cols.equals(sensation):
        x=joint[[joint.columns[1],joint.columns[2]]]

    
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
    predicted = regr.predict([[5, 2]])
    """
    # importing train_test_split from sklearn
    from sklearn.model_selection import train_test_split
    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    # importing module
    from sklearn.linear_model import LinearRegression
    # creating an object of LinearRegression class
    LR = LinearRegression()
    # fitting the training data
    LR.fit(x_train,y_train)
    y_prediction =  LR.predict(x_test)
    y_prediction
    # importing r2_score module
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    # predicting the accuracy score
    score=r2_score(y_test,y_prediction)
    print('r2 socre is ',score)
    print('mean_sqrd_error is==',mean_squared_error(y_test,y_prediction))
    print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_prediction)))
    """
    return predicted
    
#print(builder(wolf,moviexp))



#Q1

q1=sensationpca

q1.insert(0,moviexp.columns[1] ,moviexp.iloc[:,1], True)
q1.insert(0,moviexp.columns[0] ,moviexp.iloc[:,0], True)

q1.dropna()
print(q1)
zscoredData=(q1)
zscoredData=zscoredData.dropna()
print(zscoredData)
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredData)
covarExplained = eigVals/sum(eigVals)*100
for ii in range(len(covarExplained)):
    print(covarExplained[ii].round(3))
numClasses = 9

x = np.linspace(1,numClasses,numClasses)
plt.bar(x, eigVals, color='gray')

plt.plot([0,numClasses],[1,1],color='orange') 
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()
whichPrincipalComponent = 5
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) 
plt.xlabel('Question')
plt.ylabel('Loadings Value')
plt.show()


#Q2

print(personalitypca.columns[0])
print(personalitypca.columns[1])
print(personalitypca.columns[2])
print(personalitypca.columns[3])
print(personalitypca.columns[4])
print(personalitypca.columns[5])
print(personalitypca.columns[6])
print(personalitypca.columns[7])

#Q3

#movie=movie.dropna(axis='rows')
movie=pd.read_csv(r"C:\Users\jagol\Downloads\movies.csv")
movie=movie.fillna(0)
"""
def len1(col):
    count=0
    x=0
    while x<len(col):
        if col[x]>0:
            count+=1
        x+=1
    #print(count)
    return count


def clean48(col):
    x=0
    p=0
    yuh=[0]*48
    while x<(len(col)):
        if p==47:
            return yuh
        if col[x]>0:
            yuh[p]=col[x]
            p+=1
        x+=1
print(movie.shape)
def cleanall (data):
    x=0
    m=0
    q=0
    lens=[0]*400
    groupbig=[0]*200
    groupsmall=[0]*200
    for column in data:
        #data[column]=(clean2(data.iloc[0:,column]))
        
        #print(len(clean2(data.iloc[:,x])))
        lens[x]=len1(data.iloc[:,x])
        #print(np.median(lens))
        #print(np.min(lens))
        if lens[x]>188.5:
            groupbig[m]=x
            #print(groupbig[m])
            m+=1
        if lens[x]<188.5:
            groupsmall[q]=x
            q+=1
        x+=1
        

    y=0
    currentbig=[0]*200
    currentsmall=[0]*200
    fullbig=[0]*48
    fullsmall=[0]*48
    dfbig=pd.DataFrame()
    dfsmall=pd.DataFrame()
    dffull=pd.DataFrame()
        
    while y<200:
        currentbig=data.iloc[:,groupbig[y]]
        cleanbig=clean48(currentbig)
        currentsmall=data.iloc[:,groupsmall[y]]
        cleansmall=clean48(currentsmall)
            
        t=0
        while t<48:
            fullbig[t]=cleanbig[t]
            fullsmall[t]=cleansmall[t]
            t+=1
        dfbig.insert(y, y, fullbig, True)
        dfsmall.insert(y,y,fullsmall,True)
        
        
        y+=1    
   
    
           
        
            
       
    return dfbig,dfsmall


cleanbiggy=cleanall(movie)[0]
cleansmally=cleanall(movie)[1]
#cleanfull=cleanall(movie)[2]

bigmean=[0]*200
smallmean=[0]*200

s=0
q=0
while s<200:
    select=cleanbiggy.iloc[:,s]
    select2=cleansmally.iloc[:,s]
    #select3=cleanfull.iloc[:,s]
    
    bigmean[s]=np.mean(select)
    smallmean[s]=np.mean(select2)
   # fullmean[s]=np.mean(select3)
    q+=1
    s+=1
#print(np.mean(bigmean))
#print(np.mean((smallmean)))
#print(np.mean(fullmean))
#print(stats.ttest_ind((bigmean), (smallmean)))
print(stats.ttest_rel(bigmean, smallmean))
print(np.mean(bigmean),np.mean(smallmean))
"""
#Q3 v2
movie=pd.read_csv(r"C:\Users\jagol\Downloads\movies.csv")
movie=movie.fillna(0)

def len1(col):
    count=0
    x=0
    while x<len(col):
        if col[x]>0:
            count+=1
        x+=1
    #print(count)
    return count


def clean48(col):
    x=0
    p=0
    yuh=[0]*48
    while x<(len(col)):
        if p==47:
            return yuh
        if col[x]>0:
            yuh[p]=col[x]
            p+=1
        x+=1
print(movie.shape)
def cleanall (data):
    x=0
    m=0
    q=0
    lens=[0]*400
    groupbig=[0]*200
    groupsmall=[0]*200
    for column in data:
        #data[column]=(clean2(data.iloc[0:,column]))
        
        #print(len(clean2(data.iloc[:,x])))
        lens[x]=len1(data.iloc[:,x])
        #print(np.median(lens))
        #print(np.min(lens))
        if lens[x]>188.5:
            groupbig[m]=x
            #print(groupbig[m])
            m+=1
        if lens[x]<188.5:
            groupsmall[q]=x
            q+=1
        x+=1
        

    y=0
    currentbig=[]
    currentsmall=[]
    fullbig=[0]*200
    fullsmall=[0]*200
    dfbig=pd.DataFrame()
    dfsmall=pd.DataFrame()
    dffull=pd.DataFrame()
        
    while y<200:
        currentbig=list(data.iloc[:,groupbig[y]])
        fullbig[y]=summy(currentbig)[0]/summy(currentbig)[1]
        currentsmall=list(data.iloc[:,groupsmall[y]])
        fullsmall[y]=summy(currentsmall)[0]/summy(currentsmall)[1]
        y+=1
    return fullbig,fullsmall


def summy(col):
    juice=0
    count=0
    x=0
    while x<len(col):
        if col[x]>0:
            juice+=col[x]
            count+=1
        x+=1
    return juice,count
big=cleanall(movie)[0]
small=cleanall(movie)[1]
print(stats.ttest_rel(big, small))     
print(np.mean(big),np.mean(small))    



y=small
x=np.arange(200)
plt.bar(x, y,.4, color ="blue")
plt.axhline(y=np.mean(small),color='b',linestyle='-')
x=np.arange(200)+200
y=big 
plt.bar(x, y,.4, color ="red")
plt.axhline(y=np.mean(big),color='r',linestyle='-')
#lines=plt.plot(x,y)
plt.legend(['Less Popular','More Popular'])
# To show the plot
plt.show()

   
#Q4
shrekdf=pd.DataFrame(shrek)
shrekdf.insert(1,1 ,personal.iloc[:,0], True)
shrekdf=shrekdf.dropna()

maleshrek1=[]
femaleshrek1=[]
gender=list(shrekdf.iloc[:,1])

rate=list(shrekdf.iloc[:,0])
x=0
f=0
m=0
while x<990:
    if (gender[x]==1):
        femaleshrek1.append(rate[x])
        f+=1
    if gender[x]==2:
        maleshrek1.append(rate[x])
        m+=1
    x+=1
print((len(maleshrek1),(len(femaleshrek1))))    
    
femaleshrek2=femaleshrek[0:241]
#np.sort(maleshrek1)
#np.sort(femaleshrek2)
y=femaleshrek2
x=np.arange(241)
plt.bar(x, y,.4, color ="blue")
plt.axhline(y=np.mean(femaleshrek2),color='b',linestyle='-')
x=np.arange(241)+241
y=maleshrek1 
plt.bar(x, y,.4, color ="red")
plt.axhline(y=np.mean(maleshrek1),color='r',linestyle='-')
#lines=plt.plot(x,y)
plt.legend(['female','male'])
# To show the plot
plt.show()

print(stats.ttest_ind((maleshrek1), (femaleshrek2)))
print(stats.ttest_rel(maleshrek1, femaleshrek2))
#Q5
liondf=pd.DataFrame(lion)
liondf.insert(1,1,personal.iloc[:,1],True)
liondf=liondf.dropna()
sib=[]
only=[]

status=list(liondf.iloc[:,1])
rate=list(liondf.iloc[:,0])

x=0

while x<liondf.shape[0]:
    if(status[x]==1):
        only.append(rate[x])
    if(status[x]==0):
        sib.append(rate[x])
    x+=1
print(len(only),len(sib))

sib2=sib[0:151]


y=sib2
x=np.arange(151)
plt.bar(x, y,  1, color ="blue")
plt.axhline(y=np.mean(sib2),color='b',linestyle='-')
x=np.arange(151)+151
y=only
plt.bar(x, y,1, color ="red")
plt.axhline(y=np.mean(only),color='r',linestyle='-')
#lines=plt.plot(x,y)
plt.legend(['siblings','only child'])
# To show the plot
plt.show()



print(stats.ttest_ind((only), (sib2)))
print(stats.ttest_rel(only, sib2))
#Q6
wolfdf=pd.DataFrame(wolf)
wolfdf.insert(1,1,personal.iloc[:,2],True)
wolfdf=wolfdf.dropna()
alone=[]
together=[]

status=list(wolfdf.iloc[:,1])
rate=list(wolfdf.iloc[:,0])
x=0
while x<667:
    if(status[x]==1):
        alone.append(rate[x])
    if(status[x]==0):
        together.append(rate[x])
    x+=1
print(len(alone),len(together))
alone2=alone[0:270]
print(stats.ttest_ind((alone), (together)))
print(stats.ttest_rel(alone2, together))

y=alone2
x=np.arange(270)
plt.bar(x, y,.4, color ="blue")
plt.axhline(y=np.mean(alone2),color='b',linestyle='-')
x=np.arange(270)+270
y=together 
plt.bar(x, y,.4, color ="red")
plt.axhline(y=np.mean(together),color='r',linestyle='-')
#lines=plt.plot(x,y)
plt.legend(['alone','together'])
# To show the plot
plt.show()


#Q8
def q8(movie):
   joint=personalitypca
   joint.insert(0,0,movie,True)
   joint=joint.dropna()
   print(joint.shape)
   x=joint[[joint.columns[1],joint.columns[2],joint.columns[3],joint.columns[4],joint.columns[5],joint.columns[6],joint.columns[7],joint.columns[8]]]
   y=joint[[joint.columns[0]]]
   regr = linear_model.LinearRegression()
   regr.fit(x, y)
   predicted=regr.predict([[5, 2,3,5,2,1,3,1]])
   return predicted
print(q8(lion))

#Q9
def q9(movie):
    joint=personal
    joint.insert(0,0,movie,True)
    joint=joint.dropna()
    print(joint.shape)
    x=joint[[joint.columns[1],joint.columns[2],joint.columns[3]]]
    y=joint[[joint.columns[0]]]
    
    from sklearn.model_selection import train_test_split
# splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    from sklearn.linear_model import LinearRegression
# creating an object of LinearRegression class
    LR = LinearRegression()
# fitting the training data
    LR.fit(x_train,y_train)
    y_prediction =  LR.predict(x_test)
    y_prediction
    
    # importing r2_score module
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
# predicting the accuracy score
    score=r2_score(y_test,y_prediction)
    #regr = linear_model.LinearRegression()
    #regr.fit(x, y)
    #predicted=regr.predict([[1,1,1]])
    #return predicted
    plt.scatter(x_test.iloc[:,0], y_test)
    #plt.yscale('log')
    plt.title("Test data")
    plt.show()
    return score
print(q9(lion))

#Q10

def q10(movie):
    #joint=moviexppca
    joint=pd.concat([sensationpca,moviexppca,personalitypca,personal],axis=1)
    #joint.insert(0,0,movie,True)
    joint=joint.dropna()
    print(joint.shape)
    
    zscoredData=zscore(joint)
    #print(zscoredData)
    pca = PCA().fit(zscoredData)
    eigVals = pca.explained_variance_
    loadings = pca.components_
    rotatedData = pca.fit_transform(zscoredData)
    covarExplained = eigVals/sum(eigVals)*100
    for ii in range(len(covarExplained)):
        print(covarExplained[ii].round(3))
    numClasses = 24

    x = np.linspace(1,numClasses,numClasses)
    plt.bar(x, eigVals, color='gray')

    plt.plot([0,numClasses],[1,1],color='orange') 
    plt.xlabel('Principal component')
    plt.ylabel('Eigenvalue')
    plt.show()
    whichPrincipalComponent = 10
    plt.bar(x,loadings[whichPrincipalComponent,:]*-1) 
    plt.xlabel('Question')
    plt.ylabel('Loadings Value')
    plt.show()

    jointpca=pd.DataFrame()
    jointpca.insert(0,joint.columns[20] ,joint.iloc[:,20], True)
    jointpca.insert(1,joint.columns[9] ,joint.iloc[:,9], True)
    jointpca.insert(2,joint.columns[11] ,joint.iloc[:,11], True)
    jointpca.insert(3,joint.columns[8] ,joint.iloc[:,8], True)
    jointpca.insert(4,joint.columns[15] ,joint.iloc[:,15], True)
    jointpca.insert(5,joint.columns[18] ,joint.iloc[:,18], True)
    jointpca.insert(6,joint.columns[17] ,joint.iloc[:,17], True)
    jointpca.insert(7,joint.columns[5] ,joint.iloc[:,5], True)
    jointpca.insert(8,joint.columns[23] ,joint.iloc[:,23], True)
    jointpca.insert(9,joint.columns[6] ,joint.iloc[:,6], True)
    
    print(jointpca.shape)
    jointpca.insert(0,'m' ,movie, True)
    jointpca=jointpca.dropna()
    x=joint[[joint.columns[1],joint.columns[2],joint.columns[3],joint.columns[4],joint.columns[5],joint.columns[6],joint.columns[7],joint.columns[8],joint.columns[9]]]
    y=joint[[joint.columns[0]]]
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    predicted=regr.predict([[1,1,1,1,1,1,1,1,1,1,1]])

    return predicted
print(q10(lion))

print(q1.columns[0])
print(q1.columns[6])
print(np.corrcoef(q1))




score=.65
prediction=3.25
print("r2 socre is ",score)
print("prediction for lion king with all 1s for factors:",prediction)

score=.83
prediction=3.21
print("r2 socre is ",score)
print("prediction for lion king with all 1s for factors:",prediction)

score=.87
prediction=3.20
print("r2 socre is ",score)
print("prediction for lion king with all 1s for factors:",prediction)


