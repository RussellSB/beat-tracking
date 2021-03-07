import librosa
import os

from odf_function import detect_onsets
from ioi_clustering import ioi_clustering
from agent_voting import *
from post_process import filter_end_beats

def beatTracker(audiopath):
    # Loading file
    x, sr = librosa.load(os.path.join(audiopath))
    
    # Onsets and IOI Clustering
    onsets, odf_med = detect_onsets(x, width, threshold_onsets)
    ioi_common = ioi_clustering(onsets, n_clusters, limit_tempo) 
    
    # Agent beat voting
    max_time = len(x) / sr
    agents = agent_voting(max_time, onsets, ioi_common, bidirectional)
    estimated_beats = best_agent(agents)
    
    # Post filter
    if n_last_beats:
        estimated_beats = filter_end_beats(estimated_beats, odf_med, n_last_beats)
        
    return estimated_beats

# Example use case, and for evaluation refer to the second half of the main.ipynb file
if __name__ == '__main__':
    audiopath = '../ballroom-data/Jive/Media-103713.wav'
    threshold_onsets = 0.1
    n_clusters = 5
    limit_tempo = 20
    n_last_beats = 10
    bidirectional = True

    # method call
    estimated_beats = beatTracker(audiopath)
    print(np.array(estimated_beats))