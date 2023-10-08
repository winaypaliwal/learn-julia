import numpy as np
import pythonic.my_math.lin_alg as mp

class bayes_classifier():
    def __init__(self, num_dims, num_classes) -> None:
        self.num_dims = num_dims
        self.num_classes = num_classes
    def set_mean_and_sigma(self, X):
        self.osigmas = {}
        for i in range(self.num_classes): self.osigmas[i] = mp.covariance(X[i]['train'].T) 

        self.means = {}
        for i in range(self.num_classes): self.means[i] = mp.mean_2d(X[i]['train'])

    def set_class_covariance_matrices(self, case=1):
        sigma = np.zeros((self.num_dims, self.num_dims))
        for i in range(self.num_classes): sigma += self.osigmas[i]
        sigma = sigma / self.num_classes
        self.sigmas = {}
        if case == 1:
            diag_sigma_value = mp.get_diag_sum(sigma) / sigma.shape[0]
            diag_sigma = diag_sigma_value * np.eye(self.num_dims)
            for i in range(self.num_classes): self.sigmas[i] = diag_sigma
        elif case == 2:
            for i in range(self.num_classes): self.sigmas[i] = sigma
        elif case == 3:
            for i in range(self.num_classes): self.sigmas[i] = self.osigmas[i] * np.eye(self.num_dims)
        elif case == 4: self.sigmas = self.osigmas

    def compute_confusion_matrix(self, X, data_split='test'):
        self.conf_mat = np.zeros((self.num_classes, self.num_classes))
        for true_class in range(self.num_classes):
            for ii in range(X[true_class][data_split].shape[0]):
                pred_class = mp.get_max_index([mp.get_discriminant(np.expand_dims(X[true_class][data_split][ii], axis=1), self.means[i], self.sigmas[i]) for i in range(self.num_classes)])
                self.conf_mat[pred_class][true_class] += 1

        return self.conf_mat
    
    def get_class_metrics(self, class_index=1):
        tp = self.conf_mat[class_index][class_index]
        tn, fp, fn = 0, 0, 0
        for i in range(self.num_classes):
            if i != class_index: 
                tn += self.conf_mat[i][i]
                fp += self.conf_mat[class_index][i]
                fn += self.conf_mat[i][class_index]
        
        metrics = {}
        metrics['tp'] = tp
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['precision'] = tp / (tp + fp)
        metrics['recall'] = tp / (tp + fn)
        metrics['F1'] =  2 * metrics['precision'] *  metrics['recall'] / (metrics['precision'] + metrics['recall'])
        return metrics
    
    def get_overall_metrics(self):
        ttp = 0
        for i in range(self.num_classes): ttp += self.conf_mat[i][i]
        total_samples = np.sum(self.conf_mat.flatten())

        self.metrics = {}
        for i in range(self.num_classes): self.metrics[i] = self.get_class_metrics(i)
        
        self.metrics['accuracy'] = ttp / total_samples
        self.metrics['mean-precision'] = np.mean(np.array([self.metrics[i]['precision'] for i in range(self.num_classes)]))
        self.metrics['mean-recall'] = np.mean(np.array([self.metrics[i]['recall'] for i in range(self.num_classes)]))
        self.metrics['mean-F1'] = np.mean(np.array([self.metrics[i]['F1'] for i in range(self.num_classes)]))
