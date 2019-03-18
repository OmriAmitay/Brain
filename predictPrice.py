from __future__ import print_function
import os
import numpy as np
import pandas as pd
import calendar
from matplotlib.pyplot import plot, draw, show

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

import matplotlib.cm as cm
import glob
import datetime
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import math

import json
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model

from scipy.stats import levy
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
def SaveJson(kc_data_price_single_hotel_sliced_dates,hotel_id,room, board,path):
    oppetunities = {}
    oppetunities['hotelMetaData'] = {
            'hotelName': 'TownHouse',
            'hotelID': hotel_id,
            'location': 'Tel-Aviv',
            'roomType': room,
            'board': board
    }
    oppetunities['arrivalDatesAndData'] = []
    arrival_dates = kc_data_price_single_hotel_sliced_dates['arrivalDate'].apply(lambda x: x.strftime('%Y-%m-%d'))
    arrival_dates = arrival_dates.unique()
    for idx, arrival_date in enumerate(arrival_dates):
        kc_data_price_single_hotel_arrival_date=kc_data_price_single_hotel_sliced_dates.loc[kc_data_price_single_hotel_sliced_dates['arrivalDate'].isin([arrival_date])]
        kc_data_price_single_hotel_arrival_date["yield_price"]=kc_data_price_single_hotel_arrival_date["yield_price"]*23
        prices = kc_data_price_single_hotel_arrival_date["yield_price"]
        data = kc_data_price_single_hotel_arrival_date[['days_before_arrival', "yield_price"]]
        data.columns = ['daysBeforeArrival', 'yield_price']
        oppetunities['arrivalDatesAndData'].append({
            'arrivalDate': str((pd.to_datetime(arrival_date) + np.timedelta64(365,'D')).date()),
            'departureDate': 'NA',
            'buyAt': prices.median(),
            'sellAt':[
                {
                'yield_price' : (math.floor(prices.median())+math.floor((prices.max()-prices.median())/4)),
                'confidance': np.random.randint(60,85),
                },
                {
                    'yield_price': (math.floor(prices.median()) + math.floor((prices.max()-prices.median())/2)),
                    'confidance': np.random.randint(40,55),
                },
                {
                    'yield_price': np.asscalar(prices.max()),
                    'confidance': np.random.randint(20,35),
                },
            ],
            'chartXaxis': 'daysBeforeArrivale',
            'chartYaxis': 'yield_price',
            #'chartData':[data.to_json(orient='records')[1:-1].replace('},{', '} {')]
            #'chartData': data.to_json(orient='records')
            'chartData': data.to_dict('records')

        })
    if 0:
        file_Name=path + '/' + str(hotel_id) + '_' + room + '_' + board + '.json'
        with open(file_Name, 'w') as outfile:
            outfile.write(json.dumps(oppetunities, sort_keys=True, indent=4, separators=(',', ': ')))
       # json.dump(oppetunities, outfile)

def test_stationarity(hotelData):
    columns = ['days_before_arrival', 'yield_price']
    data = hotelData[['days_before_arrival', 'yield_price']]
    timeseries = pd.DataFrame(data, columns=columns)
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print (dfoutput)



class PlotDestributionFunction:
    def __init__(self):
        pass
    def plot_distribution(self):

        x = range(-5, 100, 1)

        scale_range = np.arange(20.0, 100.0, 1)
        data = []
        for i in [1, 3, 5, 7, 8, 9, 10, 20, 30, 40, 50]:
            y = 2+10 * levy.cdf(x, loc=0, scale=i)
            trace = go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='scale={}'.format(i)
            )
            data.append(trace)

        # py.sign_in(Config.SecretKeys.plotly_username(), Config.SecretKeys.plotly_password())

        figure = go.Figure(data=data)

        offline.plot(figure, output_type='file', auto_open=False, filename=os.path.join('/home/TRAX/pauly/Desktop/result', 'distribution.html'))


    # app_name = "PlotPerfectProbeProductionResults"
    # LoggerInitializer.init(app_name)


def loadYieldData(path,src_name, data_preprocessing,summary_file_name,hotel_id, room, board):
    if data_preprocessing:

        file_name = path +"\\"+ src_name
        #df = pd.read_csv(file_, index_col=None,encoding='utf-8', header=0)
        df = pd.read_excel(file_name)
        df.rename(columns={ 'c_new_data': 'yield_price'}, inplace=True)

        ######################################
        yield_to_price_file_name = path +"\\yield_to_dollars_"+ src_name+"x"
        try:
            df_yield_to_price = pd.read_excel(yield_to_price_file_name)
            df = df.merge(df_yield_to_price,how='inner',on=['yield_price'])
        except:
            print("Yeild to Dollar file is missing")
        #kc_data_price_original = kc_data_price_original.merge(df[['roomName', 'Room']], on=['roomName'])
        ######################################


        #kc_data_price_original = df.drop_duplicates(subset=['_id'], keep='first')
        kc_data_price_original=df
        kc_data_price_original['arrivalDate'] = kc_data_price_original['c_date'].dt.strftime('%Y-%m-%d')#.dt.date
        kc_data_price_original['queryDate'] = pd.Series(pd.to_datetime(kc_data_price_original['c_timestamp'], unit='ms'), kc_data_price_original.index)
        kc_data_price_original['queryDate'] = pd.Series(kc_data_price_original['queryDate'].dt.date,kc_data_price_original.index)  # Only date without time
        kc_data_price_original['days_before_arrival'] = pd.to_datetime(kc_data_price_original['arrivalDate']) - pd.to_datetime(kc_data_price_original['queryDate'])
        kc_data_price_original['days_before_arrival'] = kc_data_price_original['days_before_arrival'] / np.timedelta64(1, 'D')
        kc_data_price_original['days_before_arrival'] = pd.to_numeric(kc_data_price_original['days_before_arrival'])
        kc_data_price_original['days_before_arrival'] = kc_data_price_original['days_before_arrival'].astype('float64')

        kc_data_price_original['hotelId']=hotel_id
        kc_data_price_original['roomName'] = 'standart'
        kc_data_price_original['Room'] = room
        kc_data_price_original['board'] = board
        kc_data_price_original['priceCurrency'] = 'USD'

        kc_data_price_original.to_csv(path + '/yield_' + summary_file_name + '.csv')
    else:
        kc_data_price_original = pd.read_csv(path + '/yield_' + summary_file_name + '.csv')
    print(kc_data_price_original.shape)
    return [df_yield_to_price, kc_data_price_original]



def loadHotelsData(path,data_preprocessing,summary_file_name):
    if data_preprocessing:
        allFiles = glob.glob(path + "/hotels*.csv")
        frame = pd.DataFrame()
        list_ = []
        for file_ in allFiles:
            df = pd.read_csv(file_, index_col=None, header=0)
            list_.append(df)
        frame = pd.concat(list_)

        kc_data_price_original = frame.drop_duplicates(subset=['_id'], keep='first')
        kc_data_price_original.to_csv(path + summary_file_name)

        kc_data_price_original['queryDate'] = pd.Series(pd.to_datetime(kc_data_price_original['queryDateUTC'], unit='ms'), kc_data_price_original.index)
        kc_data_price_original['queryDate'] = pd.Series(kc_data_price_original['queryDate'].dt.date,kc_data_price_original.index)  # Only date without time
        kc_data_price_original['days_before_arrival'] = pd.to_datetime(kc_data_price_original['arrivalDate']) - pd.to_datetime(kc_data_price_original['queryDate'])
        kc_data_price_original['days_before_arrival'] = kc_data_price_original['days_before_arrival'] / np.timedelta64(1, 'D')
        kc_data_price_original['days_before_arrival'] = pd.to_numeric(kc_data_price_original['days_before_arrival'])
        hotels_IDs=kc_data_price_original['hotelId']
        unique_hotels_IDs=hotels_IDs.drop_duplicates()
        #for id in unique_hotels_IDs:
          #  kc_data_price_single_hotel = kc_data_price_original.loc[kc_data_price_original['hotelId'].isin([id])]
            #kc_data_price_single_hotel.to_csv(path + '/' + str(id)+'_' + summary_file_name + '.csv')

        kc_data_price_original.to_csv(path + '/' + summary_file_name + '.csv')
    else:
        kc_data_price_original = pd.read_csv(path + '/' + summary_file_name + '.csv')
        # kc_data_price_original['queryDate']=pd.Series(pd.to_datetime(kc_data_price_original['queryDateUTC'],unit='ms'),kc_data_price_original.index)
        # kc_data_price_original['queryDate'] = pd.Series(kc_data_price_original['queryDate'].dt.date,kc_data_price_original.index ) #Only date without time
        # kc_data_price_original['days_before_arrival'] = pd.to_datetime(kc_data_price_original['arrivalDate']) -  pd.to_datetime(kc_data_price_original['queryDate'])
        # kc_data_price_original['days_before_arrival']=kc_data_price_original['days_before_arrival']/ np.timedelta64(1, 'D')
        # kc_data_price_original['days_before_arrival']=pd.to_numeric(kc_data_price_original['days_before_arrival'])
    #print(kc_data_price_original.shape)
    # print(kc_data_price_original.describe())
    return kc_data_price_original

def addUniquRoomID(kc_data_price_original,path,hotel_id):
    # file= path + "/rooms_mapper.csv"
    file_name = path + "/rooms_mapper.xlsx"
    df = pd.read_excel(file_name, sheet_name=str(hotel_id))
    # df = pd.read_csv(file, index_col=None, header=0,error_bad_lines=False)
    kc_data_price_original = kc_data_price_original.merge(df[['roomName', 'Room']], on=['roomName'])
    kc_data_price_original.to_csv(path + "/final_with_mapper.csv")
    return kc_data_price_original

def sliceSpecificHotel(kc_data_price_original,hotel_id,Room,board):
    kc_data_price_single_hotel = kc_data_price_original.loc[kc_data_price_original['hotelId'] == hotel_id]  # Specific hotel
    print(kc_data_price_single_hotel.shape)
    kc_data_price_single_hotel = kc_data_price_single_hotel.loc[kc_data_price_single_hotel['Room'].isin([Room])]  # room type
    print(kc_data_price_single_hotel.shape)
    kc_data_price_single_hotel = kc_data_price_single_hotel.loc[kc_data_price_single_hotel['board'].isin([board])]  # board
    print(kc_data_price_single_hotel.shape)
    kc_data_price_single_hotel = kc_data_price_single_hotel.loc[kc_data_price_single_hotel['priceCurrency'].isin(['USD'])]  # currency
    return kc_data_price_single_hotel



def sliceDatesForSingleHotel(kc_data_price_single_hotel,hotel_id,room,board,path,First_date,Last_date):

    kc_data_price_single_hotel['arrivalDate']=kc_data_price_single_hotel['arrivalDate'].apply(pd.to_datetime)
    kc_data_price_single_hotel_Truncated = kc_data_price_single_hotel[kc_data_price_single_hotel['arrivalDate'].between(First_date, Last_date, inclusive=True)]
    return kc_data_price_single_hotel_Truncated

def showTrends(kc_data_price_single_hotel_sliced_dates,days_step):

    #******Draw box plot********
    fig = plt.figure()
    #kc_data_price_single_hotel_sliced_dates=kc_data_price_single_hotel_sliced_dates.iloc[::days_step, :]
    ax=sns.boxplot(x=kc_data_price_single_hotel_sliced_dates['arrivalDate'].dt.date, y=kc_data_price_single_hotel_sliced_dates['yield_price'],data=kc_data_price_single_hotel_sliced_dates)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    #plt.show()
    #***************************

    # ******Draw yield_price plot********
    fig = plt.figure()
    host = host_subplot(111)
    plt.xlabel('Days before Arrival')
    plt.ylabel('Price')
    arrival_dates=kc_data_price_single_hotel_sliced_dates['arrivalDate'].apply(lambda x: x.strftime('%Y-%m-%d'))
    arrival_dates=arrival_dates.unique()
    arrival_dates=arrival_dates[::days_step]
    colors = iter(cm.rainbow(np.linspace(0, 1, len(arrival_dates))))
    for idx,arrival_date in enumerate(arrival_dates):
        kc_data_price_single_hotel_arrival_date=kc_data_price_single_hotel_sliced_dates.loc[kc_data_price_single_hotel_sliced_dates['arrivalDate'].isin([arrival_date])]
        plt.plot(kc_data_price_single_hotel_arrival_date['days_before_arrival'],kc_data_price_single_hotel_arrival_date["yield_price"], marker="*", label=str(arrival_date), zorder=10 * idx, linewidth=1.9, alpha=0.9)  # on top
    plt.legend(loc='upper right')
    plt.show()

    # ***************************

def keepLowestPriceOnly(hotel_data,path):
    #Works on single hotel
    hotels_data_lowest_price_only=pd.DataFrame()
    arrival_dates_non_sort = hotel_data['arrivalDate'].unique()
    arrival_dates_sort = np.sort(arrival_dates_non_sort)
    arrival_dates = arrival_dates_sort
    for idx, arrival_date in enumerate(arrival_dates):
        arrival_date = str(arrival_date)
        hotels_data_specific_arrival_date = hotel_data.loc[hotel_data['arrivalDate'].isin([arrival_date])]
        hotels_data_specific_arrival_date.sort_values('yield_price', inplace=True)
        hotels_data_specific_arrival_date_no_duplicates = hotels_data_specific_arrival_date.drop_duplicates(subset=['days_before_arrival'], keep='first')
        hotels_data_specific_arrival_date_no_duplicates.sort_values('days_before_arrival', inplace=True)
        #hotels_data_specific_arrival_date_no_duplicates.to_csv(path + '/' + arrival_date + '.csv')
        hotels_data_lowest_price_only=hotels_data_lowest_price_only.append(hotels_data_specific_arrival_date_no_duplicates)

    #hotels_data_lowest_price_only=hotel_data.groupby(['arrivalDate', 'days_before_arrival']).min()
    #hotels_data_lowest_price_only=hotel_data.groupby(['arrivalDate', 'days_before_arrival'])['yield_price'].min()
    return hotels_data_lowest_price_only

# def numberOfHits(kc_data_price_single_hotel_sliced_dates,value):
#     dates = kc_data_price_single_hotel_sliced_dates['arrivalDate'].loc[kc_data_price_single_hotel_sliced_dates['yield_price'] >= value]
#     numberOfdays = dates.unique().size
#     print (numberOfdays)


def numberOfHits(data,min_value,max_value,not_valid_period=0):
    data['<= min_value'] = data['yield_price'] <= min_value
    data['>= max_value'] = data['yield_price'] >= max_value
    min_time = data[data['<= min_value']].groupby('arrivalDate')['days_before_arrival'].agg('max').reset_index()[
        ['arrivalDate', 'days_before_arrival']]
    max_time = data[data['>= max_value']].groupby('arrivalDate')['days_before_arrival'].agg('min').reset_index()[
        ['arrivalDate', 'days_before_arrival']]
    min_time = min_time.rename(index=str, columns={"days_before_arrival": "min time"})
    max_time = max_time.rename(index=str, columns={"days_before_arrival": "max time"})
    df1 = data.merge(min_time, on='arrivalDate')
    df1 = df1.merge(max_time, on='arrivalDate')
    sum_of_hits = len(df1[df1['min time'] > df1['max time']]['arrivalDate'].unique())
    return sum_of_hits


def pickOppetunities(data, yield_to_price):
    # group = kc_data.groupby('arrivalDate')
    # kc_new_data=kc_data.groupby('arrivalDate').filter(lambda x: (x['yield_price'].quantile(0.8) - x['yield_price'].quantile(0.2)) > 3)
    # return kc_new_data
    max_value=0
    buying_value = 0
    selling_value = 10000000000
    price_range = np.arange(4,20,1)
    for min_price in price_range:
        max_price_range = np.arange(min(min_price+2,20),min(min_price+4,20),1)
        for max_price in max_price_range:
            hits=numberOfHits(data, min_price, max_price)
            try:
                value = hits* (yield_to_price.loc[yield_to_price['yield_price']==max_price,'price'].iloc[0]-yield_to_price.loc[yield_to_price['yield_price']==min_price,'price'].iloc[0])
            except:
                value = 0
            if value > max_value:
                max_value = value
                buying_value = min_price
                selling_value = max_price

    return [buying_value, selling_value]


def removeOutlires(kc_data_price_original):
    kc_data_price_original.drop(kc_data_price_original[kc_data_price_original.yield_price >= 15].index, inplace=True)
    kc_data_price_original.drop(kc_data_price_original[kc_data_price_original.yield_price == 1].index, inplace=True)
    kc_data_price_original.drop(kc_data_price_original[kc_data_price_original.days_before_arrival <= 5].index,inplace=True)
    kc_data_price_original.drop(kc_data_price_original[kc_data_price_original.days_before_arrival >= 200].index,inplace=True)

    return kc_data_price_original


def simulateRevenue(data,buying_price,sell_price):
    return  numberOfHits(data, buying_price, sell_price)

    ############remove all the prices *before* the buying day for each case###

    #extract all the days that we whould had sell (out of the ones that we buy)
    pass

def add_months(sourcedate,months):
     month = sourcedate.month - 1 + months
     year = sourcedate.year + month // 12
     month = month % 12 + 1
     day = min(sourcedate.day,calendar.monthrange(year,month)[1])
     return datetime.date(year,month,day)


def main():
    path = r'C:\Users\user\Downloads\Hot\11_09_2018'  # use your path
    src_name= "TLV888.xls"
    hotel_id = 10
    summary_file_name = "uniqe_final"
    data_preprocessing = 1
    yield_analysis = 1
    First_date_of_year = datetime.datetime.strptime('01012018', "%d%m%Y").date()
    columns = ['sell_price', 'buy_price', 'hits', 'Value in Dollar']
    results = pd.DataFrame(columns=columns)


        #Last_date = datetime.datetime.strptime('01022017', "%d%m%Y").date()
        #revenue_simulation_first_day = datetime.datetime.strptime('01012018', "%d%m%Y").date()
        #revenue_simulation_last_day = datetime.datetime.strptime('01022018', "%d%m%Y").date()
        #if yield_analysis:

    room = 'standart'
    board = 'BB'
    [yield_to_price, kc_data_price_original]=loadYieldData(path,src_name, data_preprocessing,summary_file_name,hotel_id, room, board)
    kc_data_price_single_hotel = sliceSpecificHotel(kc_data_price_original, hotel_id, room, board)
    hotel_data_lowest_price_only = keepLowestPriceOnly(kc_data_price_single_hotel, path)  # to do: replace with groupby
    kc_data_price_single_hotel = removeOutlires(hotel_data_lowest_price_only)  # Need to check if works on single hotel only and if not to move up
    for month_ind in range(6,12,1):
        First_date = add_months(First_date_of_year, month_ind)
        Last_date = add_months(First_date, 1)
        revenue_simulation_first_day = add_months(First_date, 12)
        revenue_simulation_last_day = add_months(revenue_simulation_first_day, 1)
        # else:
        #     hotel_id = 316242
        #     # room = 'double room'
        #     room = 'Classic_Room'
        #     board = 'BB'
        #     kc_data_price_original=loadHotelsData(path, data_preprocessing, summary_file_name)
        #     kc_data_price_original=addUniquRoomID(kc_data_price_original,path,hotel_id)

        #test_stationarity(kc_data_price_single_hotel)
        kc_data_price_single_hotel_sliced_dates = sliceDatesForSingleHotel(kc_data_price_single_hotel,hotel_id,room,board,path,First_date,Last_date)
        #kc_data_price_single_hotel_sliced_dates.to_pickle(path+ "/hotel_data_"+str(hotel_id)+ ".pkl")
        showTrends(kc_data_price_single_hotel_sliced_dates,4)
        [buying_value, selling_value]=pickOppetunities(kc_data_price_single_hotel_sliced_dates, yield_to_price)
        #SaveJson(kc_Oppetunities,hotel_id,room, board,path)
        # value=10
        # numberOfHits(kc_data_price_single_hotel_sliced_dates,value)
        hotel_data_for_siulation = sliceDatesForSingleHotel(kc_data_price_single_hotel, hotel_id, room,board, path, revenue_simulation_first_day, revenue_simulation_last_day)
        hits = simulateRevenue(hotel_data_for_siulation,buying_value,selling_value)
        value = hits * (yield_to_price.loc[yield_to_price['yield_price'] == selling_value, 'price'].iloc[0] -yield_to_price.loc[yield_to_price['yield_price'] == buying_value, 'price'].iloc[0])
        results.loc[calendar.month_abbr[month_ind+1]]=[selling_value,buying_value, hits, value]
       # print(value)
    # p = PlotDestributionFunction()
    # p.plot_distribution()
    results.to_csv(path + "\\rvenueSimulation_hotelID=" +str(hotel_id)+ "_.csv")


if __name__ == "__main__":
    main()
