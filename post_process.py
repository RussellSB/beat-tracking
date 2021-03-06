# Removes the last n beats if they fall on silence
def filter_end_beats(estimated_beats, odf_med, n):
    
    clean_beats = estimated_beats[:-n]
    sf_prev = 0
    for b in estimated_beats[-n:]:
        sf = odf_med[int((b * sr) / hop_length)]
        if sf != 0:
            clean_beats.append(b)
            
    return clean_beats  