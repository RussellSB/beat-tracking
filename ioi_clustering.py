def ioi_clustering(onsets, n_clusters=5, limit_tempo=None, plot=False):
    
    # Compute all possible inter onset intervals
    ioi = []
    for i in range(1, len(onsets)):
        for j in range(i, len(onsets), i):
            ioi_instance = onsets[j] - onsets[j-i]

            # Limits possible iois by tempo if range is specified
            if limit_tempo:
                tempo = 60/ioi_instance
                if tempo >= limit_tempo[0] and tempo <= limit_tempo[1]:
                    ioi.append(ioi_instance)
            else:
                ioi.append(ioi_instance)

    temp = np.array(ioi).reshape(-1,1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(temp)
    
    # Construct clusters array and sort for convinience
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(ioi[i])
    clusters = sorted(clusters)
    
    if plot:
        plt.figure(figsize=(14, 3))
        plt.title('IOI Clustered Histograms')
        plt.xlabel('Inter Onset Interval (s)')
        plt.ylabel('Count')
        plt.hist(clusters, 100)  
    
    # Find the mode of each cluster
    ioi_common = []
    for clust in clusters:
        dict_counter = Counter()
        for x  in clust:
            dict_counter[round(x, 4)] += 1
    
        ioi_common.append(dict_counter.most_common()[0][0])
        
    return ioi_common
    
ioi_common = ioi_clustering(onsets, n_clusters= 5, limit_tempo=(20, 220), plot=True)  # have the option to limit tempo
ioi_common