# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:00:56 2018

@author: Denis Baskan - 878 571

Crawl data from weltfussball.de from the past 10 years. This will be 3 European, 3 World cups and several friendly matches. 
The crawlings for the cups are seperated, because the URLs are slightly different. 
Data of WC 2018 are slightly different stored to access them easily.
Qualifying and other friendly matches will be concatenated to the crawled data, 
because the websites are down.

"""

import pandas as pd
import numpy as np
import time
#import pdb #for compiling
from functions import *

#data frame to store all matches with score, year and kind of match
df = pd.DataFrame()
df['n1'] = ''
df['n2'] = ''
df['s1'] = 0
df['s2'] = 0
df['year'] = 0
df['kind'] = ''

#measure times
start = time.time()

#crawl past matches and store them in dataframe
crawl_cups(df)      #European / World cups
crawl_fmatches(df)  #friendly matches
crawl_wc18(df)      #World cup 2018
load_others(df)     #load qualifiying matches 
print("\nDownloads have been finished.",time.time()-start)

#%%

#replace umlaute
umlaute(df)

#convert to integer
df[['s1','s2','year']] = df[['s1','s2','year']].astype('int')

#drop redundant matches ()
df = df[~df[['n1','n2','s1','s2','year']].duplicated()]

#delete nations with less than n matches
del_unimp(df,10)

#save as csv-file
df.to_csv(path_or_buf = 'soccer_results_all.csv',sep = ';', index = False)
df[df['kind'] == "wm2018"].to_csv(path_or_buf = 'soccer_results_wc18.csv',sep = ';', index = False)
df[df['kind'] != "wm2018"].to_csv(path_or_buf = 'soccer_results.csv',sep = ';', index = False)



