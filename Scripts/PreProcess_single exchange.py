'''
    Script to pre-process bitcoin price
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import the dataset and encode the date

#data = pd.read_csv('data/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv')
data = pd.read_csv('data/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv')

# check if there is null data
print('Total null open prices: %s' % data['Open'].isnull().sum())
print('Total null high prices: %s' % data['High'].isnull().sum())
print('Total null low prices: %s' % data['Low'].isnull().sum())
print('Total null close prices: %s' % data['Close'].isnull().sum())
print('Total null volume prices: %s' % data['Volume_(BTC)'].isnull().sum())
print('Total null w.t prices: %s' % data['Weighted_Price'].isnull().sum())

# covert minute to date since we only care about daily data
data['date'] = pd.to_datetime(data['Timestamp'], unit='s').dt.date
df = pd.DataFrame()
group = data.groupby('date')
df["BTC_Open"] = group['Open'].mean()
df["BTC_High"] = group['High'].mean()
df["BTC_Low"] = group['Low'].mean()
df["BTC_Close"] = group['Close'].mean()
df["BTC_Volume"] = group['Volume_(BTC)'].sum()  # volume as in number of coins
# Volume currency is just volume coin * price
df["BTC_Weighted_Price"] = group['Weighted_Price'].mean()

#print df.head(5)


# read other market feature data
dfs = []
files = ['data/FX_Daily_12to17.csv', 'data/Interest Rate.csv', 'data/SPX data.csv', 'data/Treasury_1YR_BidAsk.csv',
         'data/Treasury_5YR_BidAsk.csv', 'data/Treasury_10YR_BidAsk.csv', 'data/Treasury_30YR_BidAsk.csv',
         'data/VIX_SP and NASDAQ_Close.csv']
for i in range(len(files)):
    data = pd.read_csv(files[i])
    data['date'] = data['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    data['date_key']=data['date']
    data = data.set_index('date_key')  # set index to be date
    dfs.append(data)

# concatenate other feature files
fi_data = dfs[0]
cols_to_drop = ['date_0']
for d in range(len(dfs)-1): #len(dfs) - 1
    fi_data = fi_data.join(dfs[d + 1], on='date', how='inner', lsuffix='_' + str(d), rsuffix='_' + str(d + 1))
    cols_to_drop.append('date_' + str(d + 1))

#drop all the extra date columns
fi_data = fi_data.drop(cols_to_drop, axis=1)


# join bitcoin data with other feature
df['date'] = df.index
data = df.join(fi_data, how='inner', lsuffix='_l', rsuffix='_r')
data = data.drop(['date_r', 'date_l'], axis=1)


delta = data["BTC_Close"] - data["BTC_Open"]
data['delta_raw'] = delta
data['delta_flg'] = (delta > 0).astype('int')

#print data.head(3)
#print data.tail(2)

data.to_csv(path_or_buf='data/data2.dat', sep=',',
            columns=data.columns.values, header=True, index=False)  # , index=True)
