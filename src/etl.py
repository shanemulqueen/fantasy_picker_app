import pandas as pd
import numpy as np
import re
import itertools
from datetime import datetime, timedelta
from pymongo import MongoClient
import json
import pymongo
import pickle
import time

class Cleaner(object):
    def __init__(self,track_data_path = 'data/track_info.csv'):
        self.track_dict,self.shape_dict,self.surface_dict,\
                self.bank_dict,self.track_int_dict=self.get_track_dict(track_data_path)
        self.loop_columns = ['Finish','High Pos','place_diff','place_diff_mid',
                    'Pass Diff','Fastest Lap','Laps Led', 'DRIVER RATING']
        self.norm_coeff = {'Finish':43,'High Pos':43,'place_diff':43,
                    'place_diff_mid':43,'Pass Diff':100,'Fastest Lap':500,
                    'Laps Led':500, 'DRIVER RATING':150}
        self.null_values = {'Finish':43,'High Pos':43,'place_diff':0,
                    'place_diff_mid':0,'Pass Diff':0,'Fastest Lap':0,
                    'Laps Led':0, 'DRIVER RATING':20}
        self.client = MongoClient()
# Access/Initiate Database
        self.db = self.client['NASCAR']

    def get_track_dict(self, file_path = 'data/track_info.csv'):
        """This is a doc string
        """
        track_dict = {'Martinsville Speedway':'Martinsville Speedway',
        'Bristol Motor Speedway':'Bristol Motor Speedway',
        'Richmond Raceway':'Richmond Raceway',
        'Richmond International Raceway':'Richmond Raceway',
        'Jeff Gordon Raceway':'ISM Raceway',
        'ISM Raceway':'ISM Raceway',
        'Dover Downs International Speedway':'Dover International Speedway',
        'Phoenix International Raceway':'ISM Raceway',
        'Dover International Speedway':'Dover International Speedway',
        'North Carolina Speedway':'North Carolina Speedway',
        'New Hampshire Motor Speedway':'New Hampshire Motor Speedway',
        'New Hampshire International Speedway':'New Hampshire Motor Speedway',
        'Darlington Raceway':'Darlington Raceway',
        'Kentucky Speedway':'Kentucky Speedway',
        'Charlotte Motor Speedway':'Charlotte Motor Speedway',
        'Chicagoland Speedway':'Chicagoland Speedway',
        'Homestead-Miami Speedway':'Homestead-Miami Speedway',
        'Las Vegas Motor Speedway':'Las Vegas Motor Speedway',
        "Lowe's Motor Speedway":'Charlotte Motor Speedway',
        'Kansas Speedway':'Kansas Speedway',
        'Texas Motor Speedway':'Texas Motor Speedway',
        'Atlanta Motor Speedway':'Atlanta Motor Speedway',
        'Sonoma Raceway':'Sonoma Raceway',
        'Infineon Raceway':'Sonoma Raceway',
        'Sears Point Raceway':'Sonoma Raceway',
        'Michigan Speedway':'Michigan International Speedway',
        'California Speedway':'Auto Club Speedway',
        'Auto Club Speedway':'Auto Club Speedway',
        'Michigan International Speedway':'Michigan International Speedway',
        'Charlotte Motor Speedway Road Course':'Charlotte Motor Speedway Road Course',
        'Watkins Glen International':'Watkins Glen International',
        'Indianapolis Motor Speedway':'Indianapolis Motor Speedway',
        'Pocono Raceway':'Pocono Raceway',
        'Daytona International Speedway':'Daytona International Speedway',
        'Talladega Superspeedway':'Talladega Superspeedway'}

        track_data = pd.read_csv(file_path)
        self.track_data = track_data
        shape_dict = {}
        surface_dict = {}
        bank_dict = {}
        track_int_dict = {}
        for i in range(len(track_data)):
            shape_dict[track_data['name'].iloc[i]] = track_data['shape'].iloc[i]
            surface_dict[track_data['name'].iloc[i]] = track_data['surface'].iloc[i]
            bank_dict[track_data['name'].iloc[i]] = track_data['turn_banking'].iloc[i]
            track_int_dict[track_data['name'].iloc[i]] = track_data['speed_rank'].iloc[i]
        return track_dict,shape_dict,surface_dict,bank_dict,track_int_dict

    def team_search(self,x):
        #used as a lambda function for a Pandas series to get the team name
        try:
            return re.search('\((?P<team>[A-Z,a-z,\w,\s]*)',x).group('team')
        except:
            return 'none'

    def get_roll(self, x, win_length):
        #lambda function
        return x.rolling(window = win_length,min_periods = 1).mean()
    def get_roll_median(self, x, win_length):
        #lambda function
        return x.rolling(window = win_length,min_periods = 1).median()

    def get_running(self,x):
        return x =='running'
    def val(self,x):
        return x

    def first_count(self,x):
        """#agg function, x is series"""
        return np.sum(x.apply(lambda x:1 if x ==1 else 0))
    def top_three_count(self,x):
        """#agg function, x is series"""
        return np.sum(x.apply(lambda x:1 if x <=3 else 0))
    def avg_abs(self,x):
        """#agg function, x is series"""
        return np.mean(np.abs(x.values))
    def fit_track_avgs(self):
        self.track_avgs_loop = self.track_avgs_loops(self.loops_data)
        self.track_avgs_race = self.track_avgs_race(self.race_data)

    def track_avgs_loops(self, loops_data):
        loop_data = loops_data.copy(deep = True)
        self.loop_agg = {'place_diff':self.avg_abs,'place_diff_mid':self.avg_abs,'Green Flag Passes':np.mean,
        'Quality Passes':np.mean,'High Pos.':[self.first_count,self.top_three_count],'DRIVER RATING':np.max,
        'first_led':np.max,'first_fastest':np.max,'first_diff':np.max,'Laps Led':np.max,
        'Fastest Lap':np.max}
        loop_data['first_led']=loop_data.apply(lambda x: x['Laps Led'] if x['Finish']==1 else 0,axis = 1)
        loop_data['first_fastest']=loop_data.apply(lambda x: x['Fastest Lap'] if x['Finish']==1 else 0,axis = 1)
        loop_data['first_diff']=loop_data.apply(lambda x: x['place_diff'] if x['Finish']==1 else 0,axis = 1)
        grp = loop_data.groupby(by=['track','race_name'])
        df2 = grp.aggregate(self.loop_agg)
        col_list = []
        for name in df2.columns:
            col_list.append(name[0]+'-'+name[1])
        grp2 = df2.groupby(by=['track']).aggregate(np.mean)
        grp2.columns = col_list
        return grp2
    def track_avgs_race(self, races_data):
        race_data = races_data.copy(deep = True)
        self.race_agg = {'avg_speed':np.mean,'pole_speed':np.mean,'margin':np.mean,'track_len':np.max,
        'lead_changes':np.mean,'pct_caution':np.mean,'fin_running':np.mean,'fallout':np.sum}
        race_data['pct_caution']=race_data.apply(lambda x: x['caution_laps']/x['laps'],axis = 1)
        race_data['fin_running']=race_data.apply(lambda x: 1 if x['Status']=='running' else 0,axis = 1)
        race_data['fallout']=race_data.apply(lambda x: 1 if ((x['St']<=10)& (x['Fin']>26)) else 0,axis = 1)
        #race_data['first_diff']=loop_data.apply(lambda x: x['place_diff'] if x['Finish']==1 else 0,axis = 1)
        grp = race_data.groupby(by=['track','race_name'])
        df2 = grp.aggregate(self.race_agg)
        col_list = []
        for name in df2.columns:
            #col_list.append(name[0]+'-'+name[1])
            col_list.append(name+'_track')
        grp2 = df2.groupby(by=['track']).aggregate(np.mean)
        grp2.columns = col_list
        return grp2

    def get_fin_percents(self, race_data, roll = 18,roll_median = 7):
        self.fin_percents = pd.DataFrame()
        for driver in race_data['Driver'].unique():
            grp = race_data[race_data['Driver']==driver]\
                            .groupby(by=['Driver','date'])
            thing3 = grp.aggregate({'Status':[self.get_running],'pct_fin':[self.val]})
            thing2 = thing3.apply(lambda x: self.get_roll(x, roll)).fillna(0).reset_index()
            thing2.columns = ['Driver','date','running_pct','pct_fin']
            thing2['pct_fin']=thing3['pct_fin'].apply(lambda x: self.get_roll_median(x,roll_median)).fillna(0).values
            self.fin_percents = pd.DataFrame.append(self.fin_percents,thing2,ignore_index = True,sort = False)
        self.fin_percents['Driver'] = self.fin_percents['Driver'].apply(lambda x:x.strip().replace('.','').replace(',',''))
    def get_fin_percent(self,x,field):
        #for accessing the most recent value for it.
        try:
            return self.fin_percents[(self.fin_percents['date']<x['date'])& \
                             (self.fin_percents['Driver']==x['Driver'])][field].iloc[-1:].values[0]
        except:
            return 0

    def get_loop_stats(self, loops_data, roll = 8):
        self.loop_columns = ['Finish','High Pos.','place_diff','place_diff_mid','Pass Diff.','Fastest Lap',
                    'Laps Led', 'DRIVER RATING']
        self.norm_coeff = {'Finish':43,'High Pos.':43,'place_diff':43,'place_diff_mid':43,'Pass Diff.':100,'Fastest Lap':500,
                    'Laps Led':500, 'DRIVER RATING':150}
        self.agg_dict = {elem:[np.sum] for elem in self.loop_columns}
        rolling_driver_stats = pd.DataFrame()
        loop_data = loops_data.copy(deep = True)
        loop_data['Fastest Lap'] = loop_data['Fastest Lap'].values/loop_data['Total Laps'].values
        loop_data['Laps Led'] = loop_data['Laps Led'].values/loop_data['Total Laps'].values
        for driver in loop_data['Driver'].unique():
            grp = loop_data[loop_data['Driver']==driver].\
                            groupby(by=['Driver','date'])
            #thing2 = grp.aggregate({'Finish':[self.val],'High Pos.':[self.val],'place_diff':[self.val],
            #                    'place_diff_mid':[self.val],'Pass Diff.':[self.val],'Fastest Lap':[self.val],
            #                    'Laps Led':[self.val],'DRIVER RATING':[self.val]})\
            self.temp_driver = driver
            thing2 = grp.aggregate(self.agg_dict).apply(lambda x: self.get_roll(x, roll)).fillna(0).reset_index()
            rolling_driver_stats = pd.DataFrame.append(rolling_driver_stats,thing2,ignore_index = True,sort = False)
        rolling_driver_stats.columns = ['Driver','date']+self.loop_columns
        return rolling_driver_stats

    def get_loop_track_stats(self, loops_data,roll = 5):
        """will gets stats at the track level too"""
        self.loop_columns = ['Finish','High Pos.','place_diff','place_diff_mid','Pass Diff.','Fastest Lap',
                    'Laps Led', 'DRIVER RATING']
        self.norm_coeff = {'Finish':43,'High Pos.':43,'place_diff':43,'place_diff_mid':43,'Pass Diff.':100,'Fastest Lap':500,
                    'Laps Led':500, 'DRIVER RATING':150}
        self.agg_dict = {elem:[np.sum] for elem in self.loop_columns}
        rolling_driver_stats = pd.DataFrame()
        loop_data = loops_data.copy(deep = True)
        loop_data['Fastest Lap'] = loop_data['Fastest Lap'].values/loop_data['Total Laps'].values
        loop_data['Laps Led'] = loop_data['Laps Led'].values/loop_data['Total Laps'].values
        for driver,track in itertools.product(loop_data['Driver'].unique(),loop_data['track'].unique()):
            grp = loop_data[(loop_data['Driver']==driver)&(loop_data['track']==track)].\
                            groupby(by=['Driver','track','date'])
            #thing2 = grp.aggregate({'Finish':[self.val],'High Pos.':[self.val],'place_diff':[self.val],
            #                    'place_diff_mid':[self.val],'Pass Diff.':[self.val],'Fastest Lap':[self.val],
            #                    'Laps Led':[self.val],'DRIVER RATING':[self.val]})\
            thing2 = grp.aggregate(self.agg_dict).apply(lambda x: self.get_roll(x, roll)).fillna(0).reset_index()
            rolling_driver_stats = pd.DataFrame.append(rolling_driver_stats,thing2,ignore_index = True,sort = False)
        try:
            rolling_driver_stats.drop(columns = 'index',inplace = True)
        except:
            pass
        new_cols = ['Driver','track','date'] +self.loop_columns
        #new_cols.sort()
        rolling_driver_stats.columns = new_cols

        return rolling_driver_stats
    def get_roll_val(self,x,field,data):
        """for accessing the most recent value for a driver & date. Slow."""
        try:
            return float(data[(data['date']<x['date'])&(data['track']==x['track'])& \
                             (data['Driver']==x['Driver'])][field].iloc[-1:].values[0])
        except:
            return float(data[field].mean())

    def get_roll_val_test2(self,x,field):
        """this will only access the loop stats variable- done for speed GOTTA GO FAST """
        try:
            return float(self.track_loop_stats[(self.track_loop_stats['date']<x['date'])& \
                             (self.track_loop_stats['Driver']==x['Driver'])][field].iloc[-1:].values[0])
        except:
            return float(self.loop_stats[field].mean())

    def get_roll_val_test(self,x,field):
        """this will only access the loop stats variable- done for speed GOTTA GO FAST """
        try:
            return float(self.loop_stats[(self.loop_stats['date']<x['date'])& \
                             (self.loop_stats['Driver']==x['Driver'])][field].iloc[-1:].values[0])
        except:
            return float(self.loop_stats[field].mean())

    def get_prior_busch(self,x, field):
        try:
            return len(self.busch_loops[(self.busch_loops['Datetime']==x['Datetime-1'] )& \
                    (self.busch_loops['Driver']==x['Driver'] )][field].values)
        except:
            return 0
    def get_prior_busch_field(self,x, field):
        try:
            return self.busch_loops[(self.busch_loops['Datetime']==x['Datetime-1']) & \
                    (self.busch_loops['Driver']==x['Driver'] )][field].values[0]
        except:
            return 43

    def copy_data(self,data):
        self.race_data = data[0].copy(deep = True)
        self.loops_data = data[1].copy(deep = True)
        self.busch_data = data[2].copy(deep = True)
        self.busch_loops = data[3].copy(deep = True)
    def clean_driver_and_track(self):
        self.race_data['Driver']=self.race_data['Driver'].apply(lambda x: x.strip().replace('.','').replace(',',''))
        self.loops_data['Driver']=self.loops_data['Driver'].apply(lambda x: x.strip().replace('.','').replace(',',''))
        self.busch_data['Driver']=self.busch_data['Driver'].apply(lambda x: x.strip().replace('.','').replace(',',''))
        self.busch_loops['Driver']=self.busch_loops['Driver'].apply(lambda x: x.strip().replace('.','').replace(',',''))

        self.race_data['track']=self.race_data['track'].map(self.track_dict).fillna('other')
        self.loops_data=self.loops_data.merge(self.race_data[['Driver','race_name','track','date','laps']],on=['Driver','race_name'])

        self.busch_data['track']=self.busch_data['track'].map(self.track_dict).fillna('other')
        self.busch_loops=self.busch_loops.merge(self.busch_data[['Driver','race_name','track','date','laps']],on=['Driver','race_name'])

        self.race_data['place_diff'] = self.race_data['St'] -self.race_data['Fin']
        self.race_data['pct_fin'] = self.race_data['Laps']/self.race_data['laps']
        self.busch_data['place_diff'] = self.race_data['St'] -self.race_data['Fin']
        self.busch_data['pct_fin'] = self.busch_data['Laps']/self.busch_data['laps']

        self.loops_data['place_diff'] = self.loops_data['Start'] -self.loops_data['Finish']
        self.loops_data['place_diff_mid'] = self.loops_data['Mid Race'] -self.loops_data['Finish']

        self.race_data['date'] =self.race_data['date'].apply(lambda x: str(x)[0:10])
        self.busch_data['date'] =self.busch_data['date'].apply(lambda x: str(x)[0:10])
        self.loops_data['date'] =self.loops_data['date'].apply(lambda x: str(x)[0:10])
        self.busch_loops['date'] =self.loops_data['date'].apply(lambda x: str(x)[0:10])

        self.busch_loops['place_diff'] = self.busch_loops['Start'] -self.busch_loops['Finish']
        self.busch_loops['place_diff_mid'] = self.busch_loops['Mid Race'] -self.busch_loops['Finish']
        self.busch_data['Datetime']= self.busch_data['date'].apply(lambda x:datetime.strptime(x[0:10],'%Y-%m-%d').date())
        self.busch_loops['Datetime']= self.busch_loops['date'].apply(lambda x:datetime.strptime(x[0:10],'%Y-%m-%d').date())

    def fit(self,data, loop_roll = 8, track_roll = 5,target_cols=[],function_cols = []):
        #provide a list of data, and optionally target column
        #and function columns using target cols as input.
        #names contained in the first element of data
        #currently defined for the singlur case, to predict NASCAR Races

        self.target_cols = target_cols
        self.function_cols = function_cols

        self.copy_data(data)
        #Do some minimal cleaning
        self.clean_driver_and_track()
        #driver for all tracks stats (uses window)
        #driver for track specific stats
        self.get_fin_percents(self.race_data)
        #self.loop_stats = self.get_loop_stats(self.loops_data, roll = loop_roll)
        #self.track_loop_stats = self.get_loop_track_stats(self.loops_data,roll = track_roll)
        self.fit_track_avgs()

        #self.busch_loop_stats = self.get_loop_stats(self.busch_loops)
        #get some other winston specific traits

        #get bush loop stats (keep it short- maybe mid race place diff, final place diff, top 15 and driver rating)
        #get driver/track performance (ie past performance specifc to the track)

        #

        #get performance in prior days race and add flag. Include st, fin, laps led, fastest laps, running

    def transform(self, race_data, target_cols = [], stats_lags = [12],track_stats_lags = [6]):
        clean_df = pd.DataFrame(race_data['Driver'].apply(lambda x: x.strip().replace('.','').replace(',','')))
        clean_df['Datetime-1']= race_data['date'].apply(lambda x:datetime.strptime(x[0:10],'%Y-%m-%d').date()-timedelta(days=1))
        clean_df['track'] = race_data['track'].map(self.track_dict)
        clean_df['date']=race_data['date'].apply(lambda x: str(x)[0:10])
        clean_df['concrete']= clean_df['track'].map(self.surface_dict).apply(lambda x: 1 if x == 'concrete' else 0)
        track_shapes = clean_df['track'].map(self.shape_dict)

        clean_df['bank'] = clean_df['track'].map(self.bank_dict)
        clean_df['track_int'] = clean_df['track'].map(self.track_int_dict)
        shape_flags = ['oval','quad','rectangle','road','tri','triangle']
        for shape in shape_flags:
            clean_df[shape]= track_shapes.apply(lambda x: 1 if x == shape else 0)

        clean_df['running_percent']= clean_df.apply(lambda x:self.get_fin_percent(x,'running_pct'),axis = 1)
        clean_df['pct_fin_lag']= clean_df.apply(lambda x:self.get_fin_percent(x,'pct_fin'),axis = 1)
        #get loop stats
        tab = self.db['driver_stats']
        cols_temp = []
        empty_row ={}
        now = time.time()
        for i in stats_lags:
            cols_temp +=[elem +'_{:02}'.format(i) for elem in self.loop_columns]
            empty_row.update({key +'_{:02}'.format(i):item for [key,item] in self.null_values.items()})
            lags = []
        for row in clean_df.iterrows():
            result = {key+'_{:02}'.format(i) for key in self.loop_columns}#{'date','Driver'}
            #result.update({key+ +'{:02}'.format(i) for key in self.loop_columns})
            result = tab.find_one({'date':{'$lt':row[1]['date']},'Driver':row[1]['Driver']},
                          result, sort=[('date',-1)])
            if result == None:
                #null_result = {}
                #null_result.update(empty_row)
                lags.append(empty_row)
            else:
                lags.append(result)
        temp_df = pd.DataFrame(lags).drop(columns = ['_id'])
        clean_df = clean_df.merge(temp_df,left_index=True,right_index=True,
                        suffixes = ('','_track'))
        print('Done with driver stats in {} seconds'.format(time.time()-now))

        tab = self.db['driver_track_stats']
        cols_temp = []
        empty_row ={}
        now = time.time()
        for i in track_stats_lags:
            cols_temp +=[elem +'_{:02}'.format(i) for elem in self.loop_columns]
            empty_row.update({key +'_{:02}'.format(i):item for [key,item] in self.null_values.items()})
            lags = []
        for row in clean_df.iterrows():
            result = {key+'_{:02}'.format(i) for key in self.loop_columns}#{'date','Driver'}
            #result.update({key+ +'{:02}'.format(i) for key in self.loop_columns})
            result = tab.find_one({'date':{'$lt':row[1]['date']},'Driver':row[1]['Driver'],'track':row[1]['track']},
                          result, sort=[('date',-1)])
            if result == None:
                #null_result = {}
                #null_result.update(empty_row)
                lags.append(empty_row)
            else:
                lags.append(result)
        temp_df = pd.DataFrame(lags).drop(columns = ['_id'])
        clean_df = clean_df.merge(temp_df,left_index=True,right_index=True,
                        suffixes = ('','_track'))
        print('Done with driver track stats in {} seconds'.format(time.time()-now))

        clean_df['raced_busch'] = clean_df.apply(lambda x:self.get_prior_busch(x,'Finish'),axis = 1)
        clean_df['busch_fin'] = clean_df.apply(lambda x:self.get_prior_busch_field(x,'Finish'),axis = 1)
        clean_df['busch_diff'] = clean_df.apply(lambda x:self.get_prior_busch_field(x,'place_diff'),axis = 1)
        clean_df['busch_rating'] = clean_df.apply(lambda x:self.get_prior_busch_field(x,'DRIVER RATING'),axis = 1)
        clean_df = clean_df.merge(self.track_avgs_loop, how = 'left',on = 'track')
        clean_df = clean_df.merge(self.track_avgs_race, how = 'left',on = 'track')
        #get prior days busch stats
        for feature in target_cols:
            clean_df[feature] = race_data[feature]
        clean_df.drop(columns = ['Datetime-1'],inplace=True)

        return clean_df

    def nn_transnorm(self, feature_df):
        norm_df = feature_df.copy(deep=True)
        self.norm_coeff = {'Finish':43,'High Pos.':43,'place_diff':86,'place_diff_mid':86,'Pass Diff.':100,'Fastest Lap':500,
                'Laps Led':500, 'DRIVER RATING':150}
        self.other_coeff = {'bank':33,'track_int':31,'busch_fin':43,'busch_diff':86,'busch_rating':150,'St':43,'year':2019}
        self.add_coeff = {'place_diff':0.5,'place_diff_mid':0.5,'Pass Diff.':0.5,'busch_diff':0.5}
        df_cols = set(feature_df.columns)
        for field in self.norm_coeff.keys():
            if (field+'_lag' in df_cols):
                norm_df[field+'_lag'] = norm_df[field+'_lag'].values/self.norm_coeff[field]
                norm_df[field+'_lag_track'] = norm_df[field+'_lag_track'].values/self.norm_coeff[field]

        for field in self.other_coeff.keys():
            if field in df_cols:
                norm_df[field] = norm_df[field].values/self.other_coeff[field]
        for field in self.add_coeff.keys():
            if (field+'_lag' in df_cols):
                norm_df[field+'_lag'] = norm_df[field+'_lag'].values+self.add_coeff[field]
                norm_df[field+'_lag_track'] = norm_df[field+'_lag_track'].values+self.add_coeff[field]
        return norm_df
    def nn_inverse_transnorm(self, feature_df):
        norm_df = feature_df.copy(deep=True)
        self.norm_coeff = {'Finish':43,'High Pos.':43,'place_diff':43,'place_diff_mid':43,'Pass Diff.':100,'Fastest Lap':500,
                'Laps Led':500, 'DRIVER RATING':150}
        self.other_coeff = {'bank':33,'track_int':31,'laps':500,'busch_fin':43,
                        'busch_diff':43,'busch_rating':150,'St':43,'year':2019}
        df_cols = set(feature_df.columns)
        for field in self.norm_coeff.keys():
            if (field+'_lag' in df_cols):
                norm_df[field+'_lag'] = norm_df[field+'_lag'].values*self.norm_coeff[field]
                norm_df[field+'_lag_track'] = norm_df[field+'_lag_track'].values*self.norm_coeff[field]
        for field in self.other_coeff.keys():
            if field in df_cols:
                norm_df[field] = norm_df[field].values*self.other_coeff[field]
        return norm_df
