# -*- coding: utf-8 -*-
from scipy.stats import entropy
from ..utils.base_utils import Queue
from .model_utils import *
from .shapelet_utils import *
from .distance_utils import *


def __static_shapelet_candidate_loss(cand, time_series_set, label, warp, num_segment, seg_length, measurement, **kwargs):
    assert seg_length == cand.shape[0] and num_segment * seg_length == time_series_set.shape[1]
    distances = np.zeros(time_series_set.shape[0], dtype=float)
    for i in range(time_series_set.shape[0]):
        distances[i] = pattern_distance_no_timing(pattern=cand, time_series=time_series_set[i], warp=warp, measurement=measurement)
    positive_distance = distances[label == 1]
    negative_distance = distances[label == 0]
    max_val, min_val = np.max(distances), np.min(distances)
    num_bins = int(max_val - min_val) + 1
    positive_norm = np.histogram(a=positive_distance, bins=num_bins, range=(min_val, max_val), density=True)[0]
    negative_norm = np.histogram(a=negative_distance, bins=num_bins, range=(min_val, max_val), density=True)[0]
    positive_norm[positive_norm == 0] = 1e-3
    negative_norm[negative_norm == 0] = 1e-3
    return -(entropy(negative_norm, positive_norm) + entropy(positive_norm, negative_norm))


def __static_shapelet_candidate_loss_factory(time_series_set, label, warp, num_segment, seg_length, measurement, **kwargs):
    def __main__(pid, args, queue):
        ret = []
        for cand in args:
            loss = __static_shapelet_candidate_loss(
                cand=cand, time_series_set=time_series_set, label=label, warp=warp, num_segment=num_segment,
                seg_length=seg_length, measurement=measurement, **kwargs
            )
            ret.append((cand, loss))
            queue.put(0)
        return ret
    return __main__


def learn_static_shapelets(time_series_set, label, K, C, warp, num_segment, seg_length, measurement, **kwargs):
    cands = generate_shapelet_candidate(time_series_set=time_series_set, num_segment=num_segment,
                                        seg_length=seg_length, candidate_size=C, **kwargs)
    parmap = ParMap(
        work=__static_shapelet_candidate_loss_factory(
            time_series_set=time_series_set, label=label, warp=warp, num_segment=num_segment, seg_length=seg_length,
            measurement=measurement, **kwargs
        ),
        monitor=parallel_monitor(msg='learning static shapelets', size=len(cands),
                                 debug=kwargs.get('debug', True)),
        njobs=kwargs.get('njobs', NJOBS)
    )
    return sorted(parmap.run(data=cands), key=lambda x: x[-1])[:K]
