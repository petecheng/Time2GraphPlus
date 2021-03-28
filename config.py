import numpy as np
from os import path, cpu_count
from time2graph.utils.base_utils import Debugger

module_path = path.dirname(path.abspath(__file__))
njobs = cpu_count()
if njobs >= 40:
    njobs = int(njobs / 2)

EQS = {
    'K': 50,
    'C': 800,
    'seg_length': 24,
    'num_segment': 21,
    # 'diff': True,
    # 'standard_scale': True
}

WTC = {
    'K': 20,
    'C': 400,
    'seg_length': 30,
    'num_segment': 30,
    # 'feat_norm': True,
    # 'softmax': True
    # 'diff': True
    # 'standard_scale': False
}


STB = {
    'K': 50,
    'C': 800,
    'seg_length': 15,
    'num_segment': 15,
}

STEALING = {
    'K': 100,
    'C': 1000,
    'seg_length': 24,
    'num_segment': 12
}

TELECOM = {
    'K': 100,
    'C': 1000,
    'seg_length': 24,
    'num_segment': 40
}

ELDERLY = {
    'K': 100,
    'C': 1000,
    'seg_length': 30,
    'num_segment': 24
}

model_args = {
    'ucr-Earthquakes': EQS,
    'ucr-WormsTwoClass': WTC,
    'ucr-Strawberry': STB,
    'stealing': STEALING,
    'telecom': TELECOM,
    'elderly': ELDERLY
}

xgb_args = {
    'ucr-Earthquakes': {
        'max_depth': 8,
        'learning_rate': 0.1,
        'scale_pos_weight': 1,
        'n_estimators': 80,
        'booster': 'gbtree'
    },
    'ucr-WormsTwoClass': {
        'max_depth': 1,
        'learning_rate': 0.1,
        'scale_pos_weight': 1,
        'n_estimators': 50,
        'booster': 'gblinear'
    },
    'ucr-Strawberry': {
        'max_depth': 8,
        'learning_rate': 0.2,
        'scale_pos_weight': 1,
        'n_estimators': 10,
        'booster': 'gbtree'
    },
    'elderly': {
        'max_depth': 8,
        'learning_rate': 0.1,
        'scale_pos_weight': 10,
        'n_estimators': 50,
        'booster': 'gbtree'
    }
}

__all__ = [
    'np',
    'path',
    'Debugger',
    'module_path',
    'njobs',
    'xgb_args',
    'model_args'
]
