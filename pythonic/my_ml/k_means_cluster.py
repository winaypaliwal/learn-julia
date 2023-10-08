import numpy as np
import multiprocessing

from ..my_math import lin_alg as mp

class k_means_cluster():
    def __init__(self, data, num_clusters=32) -> None:
        
        self.num_clusters = num_clusters
        self.data = data
        self.num_examples, self.num_dims = data.shape
        self.centres = self.initialise_cluster_centres()

        self.cluster_ids = np.empty(self.num_examples, dtype=np.uint8) #random assignment 
        assert(num_clusters < 256) #uint8 limit

        # self.distance_metric = distance_metric

    def initialise_cluster_centres(self, strategy='simple', init_centres=None):
        if init_centres:
            self.centres = init_centres
            return
        if strategy == 'random':
            indices = np.random.randint(low=0, high=self.num_examples, size=self.num_clusters)
        else: #simple
            stride = self.num_examples // self.num_clusters
            indices = [stride*i for i in range(self.num_clusters)]

        self.centres = np.empty((self.num_clusters, self.num_dims))
        for i,index in enumerate(indices): self.centres[i] = self.data[index]

    def find_my_cluster(self, x):
        min_dist = np.inf
        cid = 0
        for i, centre in enumerate(self.centres):
            curr_dist = mp.l2_norm_sqaured(x, centre)
            if curr_dist < min_dist:
                min_dist = curr_dist
                cid = i
        return cid

    def assign_cluster_ids(self, start_i=0, end_i=0):
        if end_i == 0: end_i = self.num_examples
        # assert(end_i > start_i)
        return [self.find_my_cluster(self.data[i]) for i in range(start_i, end_i)]

    def assign_cluster_ids_parallely(self):
        # self.cluster_ids = self.assign_cluster_ids() ##serial assignment
        NUM_PROCS = multiprocessing.cpu_count() - 1
        stride = self.num_examples // NUM_PROCS

        index_args = [(i*stride, (i+1)*stride) for i in range(NUM_PROCS)]
        index_args.append((self.num_examples - self.num_examples % NUM_PROCS, self.num_examples)) #last process
        
        with multiprocessing.Pool(processes = NUM_PROCS + 1) as pool:
            results = [result for result in pool.starmap(self.assign_cluster_ids, index_args)]

        self.cluster_ids = np.concatenate(results).astype(np.uint)
        
    
    def update_cluster_params(self):
        self.bins = np.zeros(self.num_clusters, dtype=np.uint)
        
        self.centres = np.zeros((self.num_clusters, self.num_dims))

        for i,cid in enumerate(self.cluster_ids):
            self.bins[cid] += 1
            self.centres[cid] += self.data[i]
        
        # self.bins[self.bins == np.zeros(self.num_clusters, dtype=np.uint8)] = 1
        self.centres /= np.expand_dims(self.bins, axis=1) #np.tile(self.bins, (2,1)).T

    def optimise_clusters(self, num_iters=256, stopping_percent=0.005):
        self.initialise_cluster_centres()
        last_bins = np.zeros(self.num_clusters, dtype=np.uint)
        stopping_criterion = self.num_examples * self.num_examples * stopping_percent
        for i in range(num_iters):
            self.assign_cluster_ids_parallely()
            self.update_cluster_params()
            # num_changing_points = np.sum(np.absolute(self.bins - last_bins))
            num_changing_points = np.sum((self.bins - last_bins)*(self.bins - last_bins))
            if num_changing_points < stopping_criterion and i > 4: break
            last_bins = self.bins
            print(self.bins, np.sum(self.bins), self.num_examples)    
            print(f'After {i+1} iterations, {num_changing_points=} and distortion_measure: ',  self.calc_distortion_measure()) 
        assert(np.sum(self.bins) == self.num_examples)
        print(f'After {i+1} iterations, {num_changing_points=} and distortion_measure: ',  self.calc_distortion_measure())
    def update_cluster_centres_parallely(self):
        pass

    def calc_distortion_measure(self):
        dist_measure = 0
        for i,x in enumerate(self.data): dist_measure += mp.l2_norm_sqaured(x, self.centres[self.cluster_ids[i]])
        return dist_measure/self.num_examples





class k_means_cluster_mn():
    def __init__(self, data, num_clusters=32) -> None:
        
        self.num_clusters = num_clusters
        self.data = data
        self.num_examples, self.num_dims = data.shape
        self.centres = self.initialise_cluster_centres()
        self.sigmas = np.tile(np.eye(self.num_dims), (num_clusters, 1, 1))
        self.cluster_ids = np.empty(self.num_examples, dtype=np.uint8) #random assignment 
        assert(num_clusters < 256) #uint8 limit

    def initialise_cluster_centres(self, strategy='simple'):
        if strategy == 'random':
            indices = np.random.randint(low=0, high=self.num_examples, size=self.num_clusters)
        else: #simple
            stride = self.num_examples // self.num_clusters
            indices = [stride*i for i in range(self.num_clusters)]

        self.centres = np.empty((self.num_clusters, self.num_dims))
        for i,index in enumerate(indices): self.centres[i] = self.data[index]

    def find_my_cluster(self, x):
        min_dist = np.inf
        cid = 0
        for i, (centre, sigma) in enumerate(zip(self.centres, self.sigmas)):
            curr_dist = mp.mn_norm_sqaured(x, centre, sigma)
            if curr_dist < min_dist:
                min_dist = curr_dist
                cid = i
        return cid

    def assign_cluster_ids(self, start_i=0, end_i=0):
        if end_i == 0: end_i = self.num_examples
        # assert(end_i > start_i)
        return [self.find_my_cluster(self.data[i]) for i in range(start_i, end_i)]

    def assign_cluster_ids_parallely(self):
        NUM_PROCS = multiprocessing.cpu_count() - 1
        
        stride = self.num_examples // NUM_PROCS

        index_args = [(i*stride, (i+1)*stride) for i in range(NUM_PROCS)]
        index_args.append((self.num_examples - self.num_examples % NUM_PROCS, self.num_examples)) #last process
        
        with multiprocessing.Pool(processes = NUM_PROCS + 1) as pool:
            results = [result for result in pool.starmap(self.assign_cluster_ids, index_args)]

        self.cluster_ids = np.concatenate(results)
    
    def update_cluster_params(self):
        self.bins = np.zeros(self.num_clusters, dtype=np.uint)
        
        # self.centres = np.zeros((self.num_clusters, self.num_dims))

        # for i,cid in enumerate(self.cluster_ids):
        #     self.bins[cid] += 1
        #     self.centres[cid] += self.data[i]
        # self.centres /= np.expand_dims(self.bins, axis=1) #np.tile(self.bins, (2,1)).T

        clustered_data = {c:[] for c in range(self.num_clusters)}
        
        for i,cid in enumerate(self.cluster_ids): clustered_data[cid].append(self.data[i])
        
        for c in range(self.num_clusters):
            arr_len = len(clustered_data[c])
            self.bins[c] = arr_len
            if arr_len < 2: continue
            arr = np.array(clustered_data[c])
            self.centres[c] = np.mean(arr, axis=0)
            arrmu = arr - self.centres[c]
            # self.sigmas[c] = a1g3.get_inverse_2x2(arrmu.T.dot(arrmu) / (arr_len - 1))
            try:
                self.sigmas[c] = mp.get_inverse_2x2(arrmu.T.dot(arrmu) / (arr_len - 1))
            except Exception as e:
                print(e)
                print(arrmu.T.dot(arrmu)/ (arr_len - 1))
                print(clustered_data[c])
                print(np.cov(arrmu.T))
                exit()

        
    def optimise_clusters(self, num_iters=1000, stopping_percent=0.05):
        self.initialise_cluster_centres()
        last_bins = np.zeros(self.num_clusters, dtype=np.uint)
        stopping_criterion = self.num_examples * stopping_percent
        for i in range(num_iters):
            self.assign_cluster_ids_parallely()
            self.update_cluster_params()  
            num_changing_points = np.sum(np.absolute(last_bins - self.bins))
            if num_changing_points < stopping_criterion: break
            last_bins = self.bins
            # print(self.bins, np.sum(self.bins))     
        print('distortion_measure: ',  self.calc_distortion_measure())
    def update_cluster_centres_parallely(self):
        pass

    def calc_distortion_measure(self):
        dist_measure = 0
        for i,x in enumerate(self.data): dist_measure += mp.mn_norm_sqaured(x, self.centres[self.cluster_ids[i]], self.sigmas[self.cluster_ids[i]])
        return dist_measure/self.num_examples
