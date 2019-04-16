import pandas as pd
import numpy as np
import re
import itertools
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score

import src.fantasy_scoring
from pulp import *

class LineupTool(object):

    def __init__(self):
        pass

    def split_by_race(self,data,random_state = 42,race_col = 'race_name',test_size = 0.25):
        """Will return a split by race. Input is a single df, returns a test and train df."""
        races = data[race_col].unique()
        train_races, test_races = train_test_split(races,test_size = test_size,random_state = random_state)
        in_test = lambda x: x in test_races
        in_train = lambda x: x in train_races
        return data[(data[race_col].apply(in_train))], data[(data[race_col].apply(in_test))]

    def merge_race_loops(self,race_data,loop_data,race_merge_cols = ['Driver','race_name'],
                            loop_merge_cols = ['Driver','race_name'],loop_cols = ['Fastest Lap']):
        """Helper to quickly merge and loop data. Returns the merged dataframe"""
        return race_data.merge(loop_data[loop_merge_cols+loop_cols],left_on = race_merge_cols,
                            right_on = loop_merge_cols,suffixes = ('','_l'))

    def get_max_score(self,df, driver_col = 'Driver', score_col = 'dk_score',salary_col = 'Salary',sum_col = 'dk_score'):
        """Takes input of df with scores/salary and returns the max score and line up."""
        drivers = [elem for elem in df[driver_col].values]
        dk_scores = [elem for elem in df[score_col].values]
        race_salaries = [elem for elem in df[salary_col].values]

        vals = []
        model = pulp.LpProblem('race',pulp.LpMaximize)
        driver_vars = pulp.LpVariable.dicts('include',
                                (driver for driver in drivers),
                                lowBound=0,
                                upBound=1,
                               cat='Integer')

        model += (pulp.lpSum([driver_vars[dr]*dk_scores[i] for i,dr in enumerate(drivers)]))
        model += (pulp.lpSum([driver_vars[dr]*race_salaries[i] for i,dr in enumerate(drivers)])) <=50000
        model += (pulp.lpSum([driver_vars[dr] for i,dr in enumerate(drivers)])) ==6
        model.solve()
        choose = []
        for var in driver_vars:
            choice = driver_vars[var].varValue
            choose.append(int(choice)==1)
        return df[choose][sum_col].sum(),df[choose]

    def get_regret(self, df,score_col = 'dk_score',proj_col = 'proj_score',
                                driver_col = 'Driver', salary_col = 'Salary'):
        """Takes input of predicted/ actual score and returns the difference."""
        max_score, max_lineup = self.get_max_score(df,score_col = score_col)
        pick_scores, pick_lineups = self.get_max_score(df,score_col = proj_col)
        return max_score-pick_scores, [max_score,pick_scores],[max_lineup,pick_lineups]


    def lineup_mod(self,pick_df,include = [],exclude = [], running = False,
                driver_col = 'Driver', score_col = 'dk_score',
                salary_col = 'Salary',sum_col = 'dk_score'):
        df = pick_df.copy(deep = True)

        df['score_mod']=df[score_col].values

        for driver in include:
            df['score_mod']=df.apply(lambda x:200 if x['Driver'] ==driver else x['score_mod'],axis = 1)
        for driver in exclude:
            df['score_mod']=df.apply(lambda x:0 if x['Driver'] ==driver else x['score_mod'],axis = 1)
        if running:
            df['score_mod']=df.apply(lambda x:x[score_col] \
                        if ((x['Status'] =='running')) else 0,axis = 1)

        score, team =self.get_max_score(df, driver_col, score_col = 'score_mod',
                                        salary_col = salary_col,sum_col = sum_col)
        return score, team.drop(columns=['score_mod'])
