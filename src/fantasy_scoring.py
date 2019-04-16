import pandas as pd
import numpy as np


def dk_score(data,st="St",fin = "Fin",led='Led',fastest = 'Fastest Lap'):
    return data[st]-data[fin]+data[fin].apply(lambda x: 46 if x <= 1 else 44-x)+ 0.25*data[led]+ 0.5*data[fastest]

#Daniel Hemric
#Matt Tifft
#BJ McLeod
#Garrett Smithley

def score_month(data,st="St",fin = "Fin",led='Led',fastest = 'Fastest Lap',lap_col = 'Laps'):
    #laps_adj = data[lap_col].max()/data[lap_col].max()
    return data[st]-data[fin]+data[fin].apply(lambda x: 46 if x == 1 else 44-x)+ 0.25*data[led]+ 0.5*data[fastest]
