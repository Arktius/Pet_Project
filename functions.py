# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:13:51 2018

@author: Admin
"""

import urllib
from bs4 import BeautifulSoup
import re  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
#from sklearn.model_selection import train_test_split
import datetime


def time_execution(f):
    def wrapped(*args, **kws):
        now = datetime.datetime.now()
        print('[' + str(now) + '] Call Function: ' + f.__name__ + '()')
        return f(*args, **kws)
    return wrapped

@time_execution
def crawl_cups(df):
    
    df = pd.DataFrame(columns=df)
    
    print() #prints empty line
    
    domain = r'http://www.weltfussball.de/alle_spiele/' #domain of matches
    cup = [] #list to store cups
    
    #sub-URLs European / World Cup
    cup.append(domain + r'em-2016-in-frankreich/')
    cup.append(domain + r'em-2012-in-polen-ukraine/')
    cup.append(domain + r'em-2008-in-oesterreich-schweiz/')
    cup.append(domain + r'wm-2014-in-brasilien/')
    cup.append(domain + r'wm-2010-in-suedafrika/')
    
    #go through all European / World Cups
    for i in range(0,len(cup)):
        
        year = int(re.search('\d+',cup[i]).group())
        kind = cup[i][39:41]
        
        #crawl website
        client = urllib.request.urlopen(cup[i])
        page = client.read()
        soup = BeautifulSoup(page,"lxml")
                
        #go through all matches in a cup
        for tr in soup.findAll('table',{"class": r"standard_tabelle"})[0].findAll('tr'):  
            td = tr.findAll('td')
            if len(td) == 8:
                score = re.findall('\d+',td[5].getText())
                df.loc[len(df)] = [td[2].getText(),td[4].getText(),score[0],score[1],year,kind]
        
        #translate into english for printing
        if kind == 'em':
            kind = kind.replace('em','EC')
        else:
            kind = kind.replace('wm','WC')
            
        print("Data of {}-{} were downloaded.".format(kind,year))
        
    return df


def crawl_fmatches(df):
    #domain of matches
    dom = r'http://www.weltfussball.de/alle_spiele/'
    
    #U21-/U20-friendly matches
    u21 = []
    
    #matches from 2008-2015
    for i in range(8,15):       
        u21.append(dom+r'u-21-h-freundschaft-20' + str("{:0>2d}".format(i)) + '/')
        u21.append(dom+r'u-20-h-freundschaft-20' + str("{:0>2d}".format(i)) + '/')
    
    #matches from 2016-2017
    for i in range(6,8):
        u21.append(dom+r'u21-h-freundschaft-201' + str(i) + '/')
        u21.append(dom+r'u20-h-freundschaft-201' + str(i) + '/')

    #go through all friendly matches(u21/u20)
    for i in range(0,len(u21)):
        
        year = int(re.findall('\d+',u21[i])[1])  
        kind = re.findall('\d+',u21[i])[0]
        
        #crawl the website
        client = urllib.request.urlopen(u21[i])
        page = client.read()
        soup = BeautifulSoup(page,"lxml")
                
        #go through all matches
        for tr in soup.findAll('table',{"class": r"standard_tabelle"})[0].findAll('tr'):  
            td = tr.findAll('td')
            if len(td) == 8:
                score = re.findall('\d+',td[5].getText())
                df.loc[len(df)] = [td[2].getText(),td[4].getText(),score[0],score[1],year,'U'+kind+'-F']
           
        print("Data of U{}-{} were downloaded.".format(kind,year))


def crawl_wc18(df):
    #wc2018
    wc18 = r'http://www.weltfussball.de/alle_spiele/wm-2018-in-russland/'
        
    #crawl data of WC 2018
    year = int(re.findall('\d+',wc18)[0])
    kind = 'wm'+str(year)
    client = urllib.request.urlopen(wc18)
    page = client.read()
    soup = BeautifulSoup(page,"lxml")
            
    #go through all matches
    for tr in soup.findAll('table',{"class": r"standard_tabelle"})[0].findAll('tr'):  
        td = tr.findAll('td')
        if len(td) == 8:
            score = re.findall('\d+',td[5].getText())
            df.loc[len(df)] = [td[2].getText(),td[4].getText(),score[0],score[1],year,kind]

    print("Data of {} were downloaded.".format('WC2018'))
    
    
'''
As I wrote this code snippets, the data of the matches were available on t-online.
Unfortunately these websites are no longer accessible, but I still have the data.
'''
def load_others(df):    
    #read data and concatenate them to existing
    df2 = pd.read_csv('qualy_and_others.csv',delimiter = ';')
    df = pd.concat([df,df2])

    ##qualifiying matches
    #qurl = [r'http://sportdaten.t-online.de/fussball/european-world-cup-qualifiers/qualifying-world-cup-2018/spielplan-tabelle/id_46_0_112018_336_0/', r'http://sportdaten.t-online.de/fussball/european-world-cup-qualifiers/qualifying-world-cup-2014/spielplan-tabelle/id_46_0_112014_336_0/', r'http://sportdaten.t-online.de/fussball/european-world-cup-qualifiers/qualifying-world-cup-2010/spielplan-tabelle/id_46_0_112010_336_0/',r'http://sportdaten.t-online.de/fussball/european-world-cup-qualifiers/qualifying-world-cup-2006/spielplan-tabelle/id_46_0_1000022_336_0/']
    #
    #for i in range(0,len(qurl)):
    #    year = qurl[i][92:94]
    #            
    #    client = urllib.request.urlopen(qurl[i])
    #    page = client.read()
    #    soup = BeautifulSoup(page,"lxml")
    #    
    #    for table in soup.findAll('table',{"class": r"table"}):  
    #        for tr in table.findAll('tr'):
    #            td = tr.findAll('td')
    #            if len(td) == 6:
    #                df.loc[len(df)] = [td[1].getText(),td[3].getText(),td[4].getText()[0],td[4].getText()[2],year,'wmqual']
    #
    #print('qualies finished',time.time()-start)
    #
    #                
    ##matches besides cups       
    #nurl = []
    #
    #for i in range(7):
    #    nurl.append(r'http://sportdaten.t-online.de/fussball/internationals/' + str(2017-i) + '-' + str(2018-i) + '/spielplan-tabelle/id_46_0_' + str(2017-i) + '_88_0/')
    #
    #for i in range(0,len(nurl)):
    #    year = nurl[i][61:63]
    #    
    #    client = urllib.request.urlopen(nurl[i])
    #    page = client.read()
    #    soup = BeautifulSoup(page,"lxml")
    #
    #    for table in soup.findAll('table',{"class": r"table"}):
    #        for tr in table.findAll('tr'):
    #            td = tr.findAll('td')
    #            if len(td) > 6:
    #                if td[4].getText()[0].isdigit():
    #                    df.loc[len(df)] = [td[1].getText(),td[3].getText(),td[4].getText()[0],td[4].getText()[2],year,'others']
    #


def umlaute(df):
    #replace umlaute
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Rumänien', value ='Rumaenien')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Österreich', value ='Oesterreich')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Türkei', value ='Tuerkei')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Ägypten', value ='Aegypten')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Färöer', value ='Faeroeer')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Dänemark', value ='Daenemark')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Südkorea', value ='Suedkorea')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Südafrika', value ='Suedafrika')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Elfenbeinküste', value ='Elfenbeinkueste')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Weißrussland', value ='Weissrussland')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Curaçao', value ='Curacao')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Großbritannien', value ='Grossbritannien')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'Äquatorialguinea', value ='Aequatorialguinea')
    df[['n1','n2']] = df[['n1','n2']].replace(to_replace = 'México U23', value ='Mexico U23')

    #code with lambda function
    #replace = ['Rumänien',...]
    #value = ['Rumaenien',...]
    #f = lambda replace,val: df[['n1','n2']].replace(to_replace = replace, value = val)
    #df[['n1','n2']] = f(replace,value)

def del_unimp(df,n):
    #drop unimportant nations with less than 10 matches
    nats = df['n1'].value_counts()
    nats = list(nats.index[-nats[nats < n].size:]) 
    df = df[~df['n1'].isin(nats)]
    df = df[~df['n2'].isin(nats)]


def prepro_nn(dataset):
    #x - nations, year and kind are known values
    #y - the scores are to be predicted
    x = dataset.iloc[:, [0,1,4,5]].values
    y = dataset.iloc[:, [2,3]].values
    
    #transform scores into a matrix
    #highest possible score that a player had (+1 for 0)
    hsc=dataset[['s1','s2']].max().max()+1
    
    y2 = np.zeros([len(y),hsc*2])
    
    for i in range(0,len(y2)):
        y2[i,y[i][0]] = 1
        y2[i,y[i][1]+hsc] = 1
        
    
    # Encoding categorical data
    labelencoder_x_1 = LabelEncoder()
    x[:, 0] = labelencoder_x_1.fit_transform(x[:, 0])
    labelencoder_x_2 = LabelEncoder()
    x[:, 1] = labelencoder_x_2.fit_transform(x[:, 1])
    labelencoder_x_3 = LabelEncoder()
    x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])
    
    onehotencoder = OneHotEncoder(categorical_features = [0,1,3])
    x = onehotencoder.fit_transform(x).toarray()
    x = x[:, 1:]
    
    #Splitting the dataset into the Training set and Test set
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)
         
    #for remaining matches
    dtrain = dataset[dataset['kind'] != 'wm2018']
    dtrain = dtrain[~((dtrain['kind'] == 'wm') & (dtrain['year'] == 2018)) ]
    x_train = x[dataset.iloc[dtrain.index].index]
    x_test = x[dataset[dataset['kind']== 'wm2018'].index]
    y_train = y2[dataset.iloc[dtrain.index].index]
    y_test = y2[dataset[dataset['kind']== 'wm2018'].index]
    
    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return [x_train,y_train,x_test,y_test]
    

def plot_nn(nn):
    #plot the history for accuracy                    
    plt.plot(nn.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training'], loc='upper left')
    #plt.show()
    plt.savefig('result/acc',dpi = 400)
    plt.clf()
    
    #summarize history for loss
    plt.plot(nn.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig('result/loss', dpi = 400)
    plt.clf()
    
    

'''
How good our model is will be measured in the following function. 
You get points as listed:
    
predicted score is actual score - 3 points
score tendency is correct       - 2 points
predicted the winner correctly  - 1 point

Make your own predictions and compare yourself with the model
'''
def result(fyp,fyt):
    points = 0
    for match in range(0,len(fyt),2):
        if fyp[match:match+2] == fyt[match:match+2]: 
            points += 3
        elif np.diff(fyp[match:match+2]) == np.diff(fyt[match:match+2]): 
            points += 2
        elif (fyp[match] > fyp[match+1]) == (fyt[match] > fyt[match+1]): 
            points += 1
        
    return points


#def make_pred(df):
    