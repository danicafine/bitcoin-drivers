'''
    Script to merge features by date.

    Danica Fine, Emily Jiang
'''

import numpy as np
import datetime as dt
from pandas import DataFrame, concat
from os import listdir

# define files to parse
loc = '../Data/features/'
files = listdir(loc)

# list of dataframes to concatenate
dfs = []
bc_dfs = []

# load data
for x in range(len(files)):
    bc_flg = (files[x].find('_agg.dat') != -1)  # is bitcoin file

    # get headers
    with open(loc + files[x], 'r') as f:
        headers = f.readline().replace('\n','') 

    # parse all columns into floats, skip header
    data = np.genfromtxt(loc + files[x], delimiter=',', skip_header=1)
    
    # remove rows with nan since they're useless to us
    output = data[~np.isnan(data).any(axis=1)]

    # convert to dataframe
    cols = []
    if bc_flg: 
        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
    else:
        cols = headers.split(',')
        cols[0] = 'Date'

    df = DataFrame(output, columns=cols)
    df[cols[0]] = df[cols[0]].astype(int).map(str)
    df['datekey'] = df['Date']
    df = df.set_index('datekey') # set index to be date

    # save dataframe
    if bc_flg:
        bc_dfs.append(df)
    else:
        dfs.append(df)

print 'Bitcoin files found: ',len(bc_dfs)
print 'Other files found: ',len(dfs)

# concatenate bitcoin files
bc_data = concat(bc_dfs, axis=0)
print 'Total Bitcoin datapoints: ', len(bc_data.index)

# concatenate other files
fi_data = dfs[0]
cols_to_drop = ['Date_0']
for d in range(len(dfs) - 1):
    fi_data = fi_data.join(dfs[d + 1], on='Date', how='outer', lsuffix='_'+str(d), rsuffix='_'+str(d+1))
    cols_to_drop.append('Date_'+str(d+1))

fi_data = fi_data.drop(columns=cols_to_drop)
print 'Total supplementary datapoints: ', len(fi_data.index)

# join both datasets
data = bc_data.join(fi_data, on='Date', how='inner', lsuffix='', rsuffix='_r')
data = data.drop(columns=['Date_r', '', 'Date'])

delta = data['Open'] - data['Close']
data['delta_raw'] = delta
data['delta_flg'] = (delta > 0).astype('int')

data.to_csv(path_or_buf='../Data/aligned_ftrs_exchange_rows_nodate.dat', sep=',', 
            columns = data.columns.values, header=True, index=False) #, index=True) 
