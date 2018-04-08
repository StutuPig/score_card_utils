# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:11:55 2018

@author: Peilv.Xu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
import pandas as pd
from scipy import stats
import seaborn as sns

def drawHistogram(s,num_bins=20,save=False,filename='myHist'):
    '''
    plot histogram for s
    -----------------------------------------
    Params
    s: pandas series
    num_bins: number of bins
    save: bool, is save?
    filename: png name
    -----------------------------------------
    Return
    show the plt object
    '''
    fig,ax = plt.subplots()
    mu = s.mean()
    sigma = s.std()
    n,bins,patches = ax.hist(s,num_bins,normed=1)
    
    y = mlab.normpdf(bins,mu,sigma)
    ax.plot(bins,y,'--')
    ax.set_xlabel(s.name)
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of %s: $\mu=%.2f$, $\sigma=%.2f$' % (s.name, mu, sigma))
    
    fig.tight_layout()
    if save:
        plt.savefig(filename+'.png')
        
    plt.show()

def drawPie(s,labels=None,dropna=True):
    '''
    Pie Plot for s
    -----------------------------------------
    Params
    s: pandas Series
    labels: labels of each unique value in s
    dropna:bool obj
    -----------------------------------------
    Return
    show the plt object
    '''
    counts = s.value_counts(dropna=dropna)
    if labels is None:
        labels = counts.index
    fig1,ax1 = plt.subplots()
    ax1.pie(counts,labels=labels,autopct='%1.2f%%',shadow=True,startangle=90)
    ax1.axis('equal')

    plt.show()

def drawBar(s,x_ticks=None,pct=False,dropna=False,horizontal=False):
    '''
    bar plot for s
    -------------------------------------------
    Params
    s: pandas Series
    x_ticks: list, ticks in X axis
    pct: bool, True means trans data to odds
    dropna: bool obj,True means drop nan
    horizontal: bool, True means draw horizontal plot
    -------------------------------------------
    Return
    show the plt object
    '''
    counts = s.value_counts(dropna=dropna)
    if pct == True:
        counts = counts/s.shape[0]
    ind = np.arange(counts.shape[0])
    if x_ticks is None:
        x_ticks = counts.index
    
    if horizontal == False:
        p = plt.bar(ind,counts)
        plt.ylabel('frequency')
        plt.xticks(ind,tuple(counts.index))
    else:
        p = plt.barh(ind,counts)
        plt.xlabel('frequency')
        plt.yticks(ind,tuple(counts.index))
    plt.title('Bar plot for %s' % s.name)
    
    plt.show()

def distTest(s1,s2):
    '''
    This test whether 2 samples are drawn from the same distribution
    -------------------------------------------
    Params
    s1: the first time span of one variable
    s2: the second time span of one variable
    -------------------------------------------
    Return
    t-statistics
    P-value
    '''
    return stats.ks_2samp(s1,s2)

def drawBox(inputData,horizontal=False):
    '''
    box plot for continuous variables of inputData
    -------------------------------------------
    Params
    inputData: dataset includes continuous variables
    horizontal: Orientation of the plot(Vertical or Horizontal)
    '''
    if horizontal == True:
        sns.boxplot(inputData,orient="h",palette='Set3')
    else:
        sns.boxplot(inputData,orient="v",palette='Set3')
    
def outliers_modified_z_score(s):
    '''
    detect outlier using modified z-score
    -------------------------------------------
    Params
    s: pandas series
    -------------------------------------------
    Return
    boolean results
    '''
    threshold = 3.5
    
    median = np.median(s)
    median_abs_dev = np.median(np.abs(s - median))
    modified_z_scores = 0.6745 * (s - median) / median_abs_dev
    
    return np.abs(modified_z_scores) > threshold

def outliers_iqr(s):
    '''
    detect outlier using inter quartile range
    -------------------------------------------
    Params
    s: pandas series
    -------------------------------------------
    Return
    boolean results
    '''
    quartile_1,quartile_3 = np.percentile(s,[25,75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - 1.5 * iqr
    upper_bound = quartile_3 + 1.5 * iqr
    
    return (s > upper_bound) | (s < lower_bound)

if __name__ == "__main__":
    #data1 = np.random.normal(6,1.2,1000)
    #data2 = data1 + 10
    #drawHistogram(pd.Series(data1))
    #drawHistogram(pd.Series(data2))
    #drawPie(model_data.event_source)
    #drawBar(model_data.event_source)
    #drawBox(rvs1,True)
    #print(outliers_modified_z_score(np.array(np.array([1.1,1.2,1.1,1.2,10000000]))))
    #print(outliers_iqr(np.array(np.array([1.1,1.2,1.1,1.2,200]))))

    