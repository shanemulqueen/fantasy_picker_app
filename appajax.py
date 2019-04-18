
from collections import Counter
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
from utils import object_storer
import argparse
import time
app = Flask(__name__)

from src.etl import Cleaner
from src.lineup_eval import LineupTool
import src.fantasy_scoring as fs
from collections import defaultdict


def dict_to_html(d):
    return '<br>'.join('{0}: {1}'.format(k, d[k]) for k in sorted(d)[0:100])

def table_to_dict(table,h_row=0):
    table_dict = {}
    col = []
    for elem in table[h_row].text.strip().split('\n'):
        col.append((elem,[]))
    for ind in range(h_row+1,len(table)):
        i = 0
        for elem in table[ind].text.strip().split('\n'):
            try:
                elem = int(elem)
            except:
                pass
            col[i][1].append(elem)
            i +=1
    Dict = {title:column for (title,column) in col}
    return Dict

# Form page to submit text
@app.route('/')
def submission_page_default():
    return render_template('index.html',
            lineup_html = store.lineup_html,stats_html = store.stats_html,
            scrape_button = store.home_buttons[0],fit_button = store.home_buttons[1],
            predict_button = store.home_buttons[2])

# Form page to submit text
@app.route('/index.html')
def submission_page():
    return render_template('index.html',
            lineup_html = store.lineup_html,stats_html = store.stats_html,
            scrape_button = store.home_buttons[0],fit_button = store.home_buttons[1],
            predict_button = store.home_buttons[2])

@app.route('/track_avgs')
def track_page():
    return render_template('track_avgs.html',
            #        tracks_html = store.lineup_html)
            tracks_html = store.track_avgs.to_html(classes=['display','compact', 'nowrap'],table_id='track_stats',index = False))

@app.route('/scrape', methods=['POST'] )
def scrape():
    url = str(request.json['lineup_url'])
    print('got instructions')
    response = requests.get(url)
    store.bs_obj = BeautifulSoup(response.text, 'html.parser')
    store.lineup_html = str(store.bs_obj.select('div .entry-content > table ')[0])
    #store.home_buttons[0]='hidden'
    store.home_buttons[1]='visible'
    store.home_buttons[2]='hidden'
    store.stats_html=store.race_form
    store.predicted = False
    print('Did something')
    return jsonify({'content':'<p>All new content. <em>You bet!</em></p>',
                    'lineup_html':store.lineup_html,'stats_html':store.stats_html})

@app.route('/transform', methods=['POST'] )
def tranform():
    table_rows = store.bs_obj.select('div .entry-content > table > tbody > tr ')
    store.lineup_df = pd.DataFrame(table_to_dict(table_rows)).to_html(index = False, justify = 'center')
    print(type(store.lineup_df))
    temp_df = pd.DataFrame(table_to_dict(table_rows))
    keep_cols = {key:value for [key,value] in store.scrape_cols.items() if key in set(temp_df.columns)}
    store.lineup_df=temp_df.rename(columns=keep_cols)#[list(keep_cols.values)].copy(deep=True)
    store.lineup_df['Driver']=store.lineup_df['Driver'].apply(lambda x: x.strip(' #'))
    store.lineup_df['Driver_lower']=store.lineup_df['Driver'].apply(lambda x:\
            x.strip().replace('.','').replace(',','').lower())
    salary_csv = store.get_dk_salaries(str(request.json['userid']),str(request.json['password']))

    store.lineup_df = store.lineup_df.merge(salary_csv[['Driver_lower','Salary']],on=['Driver_lower'])
    store.lineup_df.drop(columns = ['Driver_lower'],inplace=True)
    store.lineup_df['date']=str(request.json['date'])
    store.lineup_df['track']=str(request.json['track'])
    store.lineup_df['laps']=int(request.json['laps'])
    store.reload_model_form()
    store.home_buttons[1]='hidden'
    store.home_buttons[2]='visible'
    store.predicted = False
    store.stats_html=store.model_form
    store.lineup_html = store.lineup_df.to_html(index = False, classes=['display','compact', 'nowrap'],table_id='lineup_table')
    return jsonify({'table_len':str(len(table_rows)),'lineup_html':store.lineup_html,
            'stats_html':store.stats_html})

@app.route('/predict', methods=['POST'] )
def predict():
    #need to fit data
    print(type(store.lineup_df))

    include2 = str(request.json['include2']).split(', ')
    exclude2 = str(request.json['exclude2']).split(', ')
    include3 = str(request.json['include3']).split(', ')
    exclude3 = str(request.json['exclude3']).split(', ')
    if not store.predicted:
        test_cols = set(store.lineup_df.columns)
        keep_cols = [col for col in store.keep_cols if col in test_cols]
        print('Keep cols')
        print(keep_cols)
        drop_cols = [col for col in store.drop_cols if col in test_cols]
        print('Drop cols')
        print(drop_cols)
        race_features = clnr.transform(store.lineup_df,keep_cols,stats_lags = [stats_i],track_stats_lags = [track_j])
        drop_cols += ['Fastest Lap_pred','Fin_pred','Led_pred','model']#
        race_features['Fastest Lap_pred']=0#
        race_features['Fin_pred']=race_features['St']#change with dk_score
        race_features['Led_pred']=0
        race_features['model']='all'
        for i in range(len(race_features)):
            driver = race_features.iloc[i,0]
            if driver in fastest_lap.keys():
                temp_df = race_features[race_features['Driver']==driver].drop(columns=drop_cols)
                try:
                    race_features['Fastest Lap_pred'].iloc[i]= fastest_lap[driver].predict(temp_df)[0]#change with dk_score
                    race_features['Fin_pred'].iloc[i]= fin_models[driver].predict(temp_df)[0]#change with dk_score
                    race_features['Led_pred'].iloc[i]= laps_led['all'].predict(temp_df)[0]#change with dk_score
                    race_features['model'].iloc[i]='driver'
                except:
                    race_features['Fastest Lap_pred'].iloc[i]=fastest_lap['all'].predict(temp_df.values.reshape(1, -1))[0]#change with dk_score
                    race_features['Fin_pred'].iloc[i]= fin_models['all'].predict(temp_df.values.reshape(1, -1))[0]#change with dk_score
                    race_features['Led_pred'].iloc[i]= laps_led['all'].predict(temp_df.values.reshape(1, -1))[0]#change with dk_score
            else:
                temp_df = race_features.drop(columns=drop_cols).iloc[i,:]
                race_features['Fastest Lap_pred'].iloc[i]=fastest_lap['all'].predict(temp_df.values.reshape(1, -1))[0]
                race_features['Fin_pred'].iloc[i]= fin_models['all'].predict(temp_df.values.reshape(1, -1))[0]
                race_features['Led_pred'].iloc[i]= laps_led['all'].predict(temp_df.values.reshape(1, -1))[0]
        #need transform the lineup and figure out keep cols
        store.forecast_df=race_features.copy(deep = True)
        store.sum_col = 'dk_score_pred'
        # if 'Fin' in store.forecast_df.columns:
        #     store.forecast_df['dk_score'] = fs.dk_score(store.forecast_df)
        #     store.sum_col = 'dk_score'
        store.forecast_df['dk_score_pred'] = fs.dk_score(store.forecast_df,
                            fin = "Fin_pred",led='Led_pred',fastest = 'Fastest Lap_pred')#change with dk_score
        test_cols = set(store.forecast_df.columns)
        pick_cols = [col for col in store.pick_cols if col in test_cols]
        store.lineup_html = store.forecast_df[pick_cols].to_html(\
                            index = False, classes=['display','compact', 'nowrap'],
                            table_id='lineup_table')
        store.predicted = True


    score,picks = lineup_tool.lineup_mod(store.forecast_df,score_col = 'dk_score_pred',sum_col= store.sum_col)
    store.lineup1 = picks.copy(deep = True)
    score2,picks2 = lineup_tool.lineup_mod(store.forecast_df,include = include2,
                        exclude = exclude2,score_col = 'dk_score_pred',sum_col= store.sum_col)
    store.lineup2 = picks2.copy(deep = True)
    score3,picks3 = lineup_tool.lineup_mod(store.forecast_df,include = include3,
                        exclude = exclude3,score_col = 'dk_score_pred',sum_col= store.sum_col)
    store.lineup3 = picks3.copy(deep = True)
    test_cols = set(store.lineup3.columns)
    pick_cols = [col for col in store.pick_cols if col in test_cols]
    store.stats_html=store.model_form
    store.stats_html += store.lineup1[pick_cols].to_html(index = False, classes=['display','compact', 'nowrap'],table_id='lineup1')
    store.stats_html += store.lineup2[pick_cols].to_html(index = False, classes=['display','compact', 'nowrap'],table_id='lineup2')
    store.stats_html += store.lineup3[pick_cols].to_html(index = False, classes=['display','compact', 'nowrap'],table_id='lineup3')
    return jsonify({'lineup_html':store.lineup_html,'stats_html':store.stats_html})



if __name__ == '__main__':
    with open('data/model_pickles/fastest_lap.pkl', 'rb') as f:
        fastest_lap=pickle.load(f)
    with open('data/model_pickles/fin.pkl', 'rb') as f:
        fin_models=pickle.load(f)
    with open('data/model_pickles/laps_led.pkl', 'rb') as f:
        laps_led=pickle.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("roll_stats", help="Window length for Driver Stats", type=str)
    parser.add_argument("roll_track_stats", help="Window length for Driver Track Stats", type=str)
    args = parser.parse_args()
    stats_i,track_j=int(args.roll_stats),int(args.roll_track_stats)
    file_path = 'data/clean_data{:02}_{:02}'.format(stats_i,track_j)

    table = ""
    store = object_storer()
    race_data = pd.read_csv('data/raw/winston_scrape.csv')
    loop_data = pd.read_csv('data/raw/winston_loops.csv')
    busch_data = pd.read_csv('data/raw/busch_scrape.csv')
    busch_loops = pd.read_csv('data/raw/busch_loops.csv')
    data =[race_data,loop_data,busch_data,busch_loops]

    clnr = Cleaner()
    clnr.fit(data)

    store.track_avgs = clnr.track_avgs_loop.merge(clnr.track_avgs_race, left_index=True,right_index=True).reset_index()
    lineup_tool = LineupTool()
    train = False
    store.predicted = False
    if train:

        features=['asdf']
        has_model = set()
        targets = ['race_name','Fin','St','Status','Led','Fastest Lap','place_diff','laps','year','pct_fin']
        drop_cols = ['Driver','track','date','race_name','Fin','Status','Led',
                     'Fastest Lap','place_diff','year','pct_fin']
        begin = time.time()

        print('Begin fitting')
        train_df = pd.read_csv(file_path)

        start = time.time()


        train_drivers = train_df['Driver'].unique()
        untrained_drivers = []
        trained_drivers = []

        for driver in train_drivers:
            x = train_df[train_df['Driver']==driver].drop(columns= drop_cols)
            if len(x)<160:
                #print('Not enough data for {}'.format(driver))
                untrained_drivers.append(driver)
            else:
                target = 'Fastest Lap'
                y =train_df[train_df['Driver']==driver][target]
                fastest_lap[driver].fit(x, y)

                target = 'Fin'
                y =train_df[train_df['Driver']==driver][target]
                fin_models[driver].fit(x, y)

                target = 'Led'
                y =train_df[train_df['Driver']==driver][target]
                laps_led[driver].fit(x, y)
                #print('Done with {}'.format(driver))
                trained_drivers.append(driver)
        driver = 'all'
        x = train_df.drop(columns= drop_cols)
        has_model = set(trained_drivers)
        features=list(x.columns)
        target = 'Fastest Lap'
        y =train_df[target]
        fastest_lap[driver].fit(x, y)
        target = 'Fin'
        y =train_df[target]
        fin_models[driver].fit(x, y)
        target = 'Led'
        y =train_df[target]
        laps_led[driver].fit(x, y)
        print('Done with training in {} seconds'.format(time.time()-begin))

    #####################################################    ##################################################

    app.run(host='0.0.0.0', port=8081, debug=True)
