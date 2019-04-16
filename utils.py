import pandas as pd
import numpy as np
from splinter import Browser
from lxml import html
import requests
import bs4
import time
import urllib
import html5lib
from contextlib import closing

class object_storer(object):
    def __init__(self):
        self.bs_obj = []
        self.stats_html = ''
        self.lineup_html = ''
        self.lineup_df = []
        self.forecast_df = []
        self.home_buttons = ['visible','hidden','hidden']
        self.predicted = False
        self.lineup1 = []
        self.lineup2 = []
        self.lineup3 = []
        self.track_avgs = []

        self.scrape_cols={'St':'St','Pos':'St','Str':'St','Pos.':'St',
                'Name':'Driver','Driver':'Driver',
                'Fin':'Fin','Finish':'Fin',
                'Status':'Status','Laps Led':'Led'}
        self.keep_cols = ['St','Driver','Salary','date','track','laps','Status','Fin',
                        'Laps Led','Fastest Lap',"dk_score",]
        self.drop_cols = ['Driver','Salary','date','track','Status','Fin',"dk_score",
                        'Laps Led','Fastest Lap']
        f = open('templates/forecast_form.html')

        self.pick_cols = ["Driver", "St", "Salary", "dk_score", "Fin", "Led",
                    "Fastest Lap", "dk_score_pred", "Fin_pred", "Led_pred", "Fastest Lap_pred",'model',
                    "Finish_02", "place_diff_02", "DRIVER RATING_02",
                    "High Pos_02", "Finish_06", "DRIVER RATING_06"]
        self.race_form = ''
        for line in f:
            self.race_form += line.strip()
        f = open('templates/model_form.html')
        self.model_form = ''
        for line in f:
            self.model_form += line.strip()

    def get_dk_salaries(self,username, password, do_web = False,filepath = '~/Downloads/DKSalaries.csv'):
        if do_web:
            br = Browser('chrome')
            br.visit('https://www.draftkings.com')
            br.find_by_id('sign-in-link').click()
            br.find_by_name('username').fill(username)
            br.find_by_name('password').fill(password)
            br.find_by_text('LOG IN').click()
            #br.find_by_text('NAS').click()
            time.sleep()
            br.find_by_text('Enter')[0].click()
            br.find_by_text('Export to CSV')[0].click()
            br.quit()
        salary_csv = pd.read_csv(filepath)
        salary_csv['Driver_lower']=salary_csv['Name'].apply(lambda x: x.strip().replace('.','').replace(',','').lower())

        return salary_csv
    def reload_model_form(self):
        f = open('templates/model_form.html')
        self.model_form = ''
        for line in f:
            self.model_form += line.strip()
