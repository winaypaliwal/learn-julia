import matplotlib.image as mpl_image
import numpy as np

from ..my_ml.k_means_cluster import k_means_cluster, k_means_cluster_mn



def calc_colour_histogram_of_a_patch(img, num_bins=8, patch_dim=32):
    h, w, num_channels = img.shape
    bins = np.zeros(num_channels*num_bins, dtype=np.uint16)
    for c in range(num_channels):
        for i in range(h):
            for j in range(w):
                bin_number = img[i][j][c] // patch_dim
                bins[c*num_bins + bin_number] += 1
    return bins

def calc_histograms_of_an_image(img_name, patch_dim=32, save=False):
    img = mpl_image.imread(img_name)
    h, w, _ = img.shape
    #ignoring edge leftovers
    list_of_hists = []
    normalising_factor = patch_dim*patch_dim
    for ii in range(0, h - h%patch_dim, patch_dim):
        for jj in range(0, w - w%patch_dim, patch_dim):
            list_of_hists.append(calc_colour_histogram_of_a_patch(img[ii:ii+patch_dim, jj:jj+patch_dim, :]) / normalising_factor)
    
    list_of_hists = np.array(list_of_hists)
    if save: np.save(img_name, list_of_hists) #will be saved as <img_name>.npy
    #np.savetxt(img_name+'.txt', list_of_hists)
    return list_of_hists

def make_bag_of_visual_words(set_of_bin24, num_clusters=32, clustering_metric='Euclidean'):
    kmc = k_means_cluster(set_of_bin24, num_clusters=num_clusters) if clustering_metric == 'Euclidean' else k_means_cluster_mn(set_of_bin24, num_clusters=num_clusters) #'Mahalanobis Metric
    kmc.optimise_clusters()
    return kmc.bins / kmc.num_examples

def calc_overlapping_patch_stats(img_name, patch_dim=7, stride=1):
    img = mpl_image.imread(img_name)
    h, w = img.shape
    features = []
    assert(stride < patch_dim)
    for i in range(0, h - h%patch_dim, stride):
        for j in range(0, w - w%patch_dim, stride):
            arr = img[i:i+patch_dim, j:j+patch_dim].flatten()
            features.append([np.mean(arr), np.var(arr)])
    return features