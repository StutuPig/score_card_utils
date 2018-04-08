import numpy as np
import pandas as pd
from scipy import stats
import math
from sklearn.utils.multiclass import type_of_target

def readFile(path):
    '''
    read raw data from a user defined file
    @Params: String 
    @Return: Pandas DataFrame
    '''
    dataset = None
    if type(path) == str and 'csv' in path:
        dataset = pd.read_csv(path)
    elif type(path) == str and 'pkl' in path:
        dataset = pd.read_pickle(path)
    elif type(path) == str and 'xlsx' in path:
        dataset = pd.read_excel(path)
    else:
        raise TypeError("Input data should be in following types: csv,pkl,xlsx.")
    
    return dataset

def split(content,i):
    '''
    Split the attributes, collect the data of the ith attribute, i = 0,1,2,3...
    Return a dataframe with selected attribute and the target
    
    notes: the target variable should be the last column
    @Params: Pandas DataFrame 
    @Return: Pandas DataFrame 
    '''
    #Test
    #content.iloc[9,3] = np.nan
    content = content.iloc[:,[i,-1]]
    
    return content

def count(data_set):
    '''
    Count the number of same record
    
    @Params: Pandas DataFrame
    @Return: Pandas DataFrame
    '''
    columns = data_set.columns.values.tolist()
    data_set['count'] = 1
    counted_data = data_set.groupby(columns,as_index=False)['count'].sum()
    
    return counted_data

def build(counted_data):
    '''
    Build a structure that ChiMerge algorithm works properly on it 
    Return a dictionary
    @Params: tuple
    @Return: list of dictionaries
    '''
    target_values = counted_data.iloc[:,-2].value_counts().index.tolist()
    n = len(target_values)
    length_dic = {}
    for record in counted_data.values.tolist():
        #print(record)
        flag = 0
        if record[0] not in length_dic.keys():
            length_dic[record[0]] = [0] * n
        for i in range(n):
            if record[1] == target_values[i]:
                length_dic[record[0]][i] = record[2]
                flag += 1
        if flag == 0:
            raise TypeError("Data Exception")
        #elif record[1] == 'Iris-virginica':
            #length_dic[record[0]][2] = record[2]  
    #print(length_dic)
    length_dic = sorted(length_dic.items())
    
    return length_dic

def Initialize(content,i):
    dataset = split(content,i)
    counted_data = count(dataset)
    length_dic = build(counted_data)
    
    return length_dic

def chi2(intervals):
    '''Compute the Chi-Square value'''      
    m=len(intervals)  
    num_class=len(intervals[0])  
    #sum of each row
    Rows=[]  
    for i in range(m):  
        sum=0  
        for j in range(num_class):  
            sum+=intervals[i][j]  
        Rows.append(sum) 
    #sum of each column
    Cols=[]  
    for j in range(num_class):  
        sum=0  
        for i in range(m):  
            sum+=intervals[i][j]  
        Cols.append(sum)  
    #total number in the intervals
    N=0  
    for i in Cols:  
        N += i  
    
    chi_value=0  
    for i in range(m):  
        for j in range(num_class):  
            Estimate=Rows[i]*Cols[j]/N  
            if Estimate!=0:  
                chi_value=chi_value+(intervals[i][j]-Estimate)**2/Estimate  
    return chi_value  

def ChiMerge(length_dic,max_interval): 
    #TODO: nan_flag
    ''' ChiMerge algorithm 
    Return split points '''    
    num_interval=len(length_dic)
    #print(length_dic)
    ceil = max(record[0] for record in length_dic) 
    print(ceil) 
    while(num_interval>max_interval):                 
        num_pair=num_interval-1  
        chi_values=[]
        #calculate the chi value of each neighbor interval  
        for i in range(num_pair): 
            intervals=[length_dic[i][1],length_dic[i+1][1]]  
            chi_values.append(chi2(intervals))  
        # get the minimum chi value 
        min_chi=min(chi_values)
        for i in range(num_pair-1,-1,-1): # treat from the last one, because I change the bigger interval as 'Merged' 
            if chi_values[i]==min_chi:
                # combine the two adjacent intervals
                temp = length_dic[i][:]
                for j in range(len(length_dic[i+1])):
                     temp[1][j] += length_dic[i+1][1][j]
                
                length_dic[i]=temp  
                length_dic[i+1]='Merged'
        while('Merged' in length_dic): # remove the merged record  
            length_dic.remove('Merged')  
        num_interval=len(length_dic)
        
    split_points = []
    for record in length_dic:
        split_points.append(record[0])
    
    print('split_point = {lst} \nfinal intervals'.format(lst = split_points))
    split_points.append(ceil)
    
    for i in range(len(split_points)-1):
        print(str(split_points[i]) + '~' + str(split_points[i+1]))

    return(split_points)  

'''
@deprecated
def detectNaN(length_dic):
    
    length_dic_bf = len(length_dic)
    nan_flag = False
    length_dic = [record for record in length_dic if record[0] != 'NaN']
    length_dic_af = len(length_dic)
    if length_dic_bf != length_dic_af:
        nan_flag = True
        
    return length_dic,nan_flag
'''

def discrete(inputData):  
    ''' ChiMerege discretization of the Iris plants database '''  
    max_interval = 6
    attribute_list = inputData.iloc[:,:-1].columns.values.tolist()
    
    #inputData = inputData.loc[:,attribute_list+target]
    intervals = {}
    for i in range(len(attribute_list)):
        print('\n'+attribute_list[i])
        #length_dic,pokemon = Initialize(pokemon,i)
        dataset = split(inputData,i)
        counted_data = count(dataset)
        length_dic = build(counted_data)
        
        split_points = ChiMerge(length_dic,max_interval)
        intervals[attribute_list[i]] = split_points

    return intervals,inputData

def dataTransfer(inputData,intervals):
    #TODO continue on reflecting
    '''
    Project the binning points into corresponding variables
    '''
    #data = inputData.loc[:,list(intervals.keys())]
    for col in intervals.keys():
        #print(intervals[col])
        inputData[col+'_bin'] = inputData[col].apply(bin_convar,values=intervals[col])
        
    return inputData

def bin_convar(x,values):
    #values = [float(item) for item in values]
    x_new = None
    for i in range(len(values)-1):
        if x <= values[i+1] and x >= values[i]:
            x_new = str(values[i]) + '-' + str(values[i+1])
            #print(x_new)
    
    return x_new

def woe_single_x(x, y, event=1, EPS=1e-7):
    """
    calculate woe and information for a single feature
    -----------------------------------------------
    Param 
    x: 1-D pandas dataframe starnds for single feature
    y: pandas Series contains binary variable
    event: value of binary stands for the event to predict
    -----------------------------------------------
    Return
    dictionary contains woe values for categories of this feature
    information value of this feature
    """
                
    check_target_binary(y)

    event_total, non_event_total = count_binary(y, event=event)
                    
    x_labels = x.unique()
    #x_labels = np.unique(x)
    woe_dict = {}
    iv = 0
    for x1 in x_labels:
                        
        y1 = y[np.where(x == x1)[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        rate_event = 1.0 * event_count / event_total#
        rate_non_event = 1.0 * non_event_count / non_event_total#
        if rate_event == 0:#
            rate_event = EPS
        elif rate_non_event == 0:#
            rate_non_event = EPS
        else:
            pass
        woe1 = math.log(rate_event / rate_non_event)#
        woe_dict[x1] = woe1
        iv += (rate_event - rate_non_event) * woe1#
            
    return woe_dict, iv

def check_target_binary(y):
    """
    check if the target variable is binary
    ------------------------------
    Param
    y:exog variable, pandas Series contains binary variable
    ------------------------------
    Return
    if y is not binary, raise a error   
    """
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('目标变量必须是二元的！')

def count_binary(a, event=1):
    """
    calculate the cross table of a
    ------------------------------
    Params
    a: pandas Series contains binary variable
    event: treate as 1, others as 0
    ------------------------------
    Return
    event_count: numbers of event=1
    non_event_count: numbers of event!=1
    """
    event_count = (a == event).sum()
    non_event_count = a.shape[-1] - event_count
    
    return event_count, non_event_count
    
def _single_woe_trans(x, y):
    """
    single var's woe trans
    ---------------------------------------
    Param
    x: single exog, pandas series
    y: endog, pandas series
    ---------------------------------------
    Return
    x_woe_trans: woe trans by x
    woe_map: map for woe trans
    info_value: infor value of x
    """
    #cal_woe = WOE()
    woe_map, info_value = woe_single_x(x, y)
    x_woe_trans = x.map(woe_map)
    x_woe_trans.name = x.name + "_WOE"
    
    return x_woe_trans, woe_map, info_value

def woe_trans(varnames, y, df):
    """
    WOE translate for multiple vars
    ---------------------------------------
    Param
    varnames: list
    y:  pandas series, target variable
    df: pandas dataframe, endogenous vars
    ---------------------------------------
    Return
    df: pandas dataframe, trans results
    woe_maps: dict, key is varname, value is woe
    iv_values: dict, key is varname, value is info value
    """
    iv_values = {}
    woe_maps = {}
    for var in varnames:
        x = df[var]
        x_woe_trans, woe_map, info_value = _single_woe_trans(x, y)
        df = pd.concat([df, x_woe_trans], axis=1)
        woe_maps[var] = woe_map
        iv_values[var] = info_value
        
    return df, woe_maps, iv_values
          

if __name__ == '__main__':
    model_data = pd.read_pickle("model_data.pkl")
    
    temp = model_data.loc[:,'ln_query_corp_cnt_lst_12m':'cl_query_self'].join(model_data.loc[:,'fspd30_recalled'])
    temp = temp.dropna(axis=0,how='any')
    
    intervals,inputData = discrete(temp.copy())
    ds = dataTransfer(inputData.copy(),intervals)
    df,woe_maps,iv_values = woe_trans(ds.loc[:,'ln_query_corp_cnt_lst_12m_bin':].columns.values.tolist()
    ,ds['fspd30_recalled'],ds)  
    
    
    '''
    pokemon = readFile("pokemon.csv")
    max_interval = 6
    attribute_list = ['HP','Attack','Defense','Speed']
    target = ['Legendary']
    pokemon = pokemon.loc[:,attribute_list+target]
    pokemon.iloc[9,3] = np.nan 
    intervals = {}
    for i in range(len(attribute_list)):
        print('\n'+attribute_list[i])
        dataset = split(pokemon,i)
        counted_data = count(dataset)
        length_dic = build(counted_data)
        #length_dic = Initialize(pokemon,i)
        #dtt_rst = detectNaN(length_dic)
        split_points = ChiMerge(length_dic,max_interval)
        intervals[attribute_list[i]] = split_points
    '''
    '''
    pokemon = pd.read_csv("pokemon.csv")
    pokemon = pokemon.loc[:,['HP','Attack','Defense','Speed','Legendary']]
    intervals,inputData = discrete(pokemon.copy())
    ds = dataTransfer(inputData.copy(),intervals)
    df,woe_maps,iv_values = woe_trans(ds.loc[:,'HP_bin':'Speed_bin'].columns.values.tolist()
    ,ds['Legendary'],ds)'''