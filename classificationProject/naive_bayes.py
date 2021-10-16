import numpy as np


# Simple Naive Bayes implementation
class NaiveBayes:

    p_y = None
    p_xy = None
    not_p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c)
        counts = np.bincount(y)
        p_y = counts / n

        # Create array to store the probability of each word for each newsgroup
        p_xy_before_transpose = np.zeros((counts.size, d))

        # Calculate the probability of word present for each newsgroup
        for label in range(counts.size):
            # Segregate each news post by its newsgroup
            x_i_for_label = X[y[:] == label, :]
            # Add all occurrence of each word by the newsgroup
            x_i_sum_for_label = x_i_for_label.sum(axis=0)
            # Divide the occurrences by the number total number of posts
            p_xy_for_label = x_i_sum_for_label / counts[label]
            # Store probabilities for corresponding newsgroup
            p_xy_before_transpose[label] = p_xy_for_label

        # Transpose result to match the implementation of predict()
        p_xy = p_xy_before_transpose.transpose()

        self.p_y = p_y
        self.p_xy = p_xy
        self.not_p_xy = 1 - p_xy

    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        not_p_xy = self.not_p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):
	    # initialize and calculate the p(y) terms
            probs = p_y.copy() 
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= not_p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred


# Naive Bayes implementation with Laplace 
class NaiveBayesLaplace(NaiveBayes):

    def __init__(self, num_classes, beta=0):
        super().__init__(num_classes)
        self.beta = beta
        
    def fit(self, X, y):
	    n, d = X.shape

	    # Compute the number of class labels
	    k = self.num_classes

	    # Compute the probability of each class i.e p(y==c)
	    counts = np.bincount(y)
	    p_y = counts / n
		
	    # Calculate the probability of word present for each newsgroup with Laplace	
	    p_xy = np.zeros((d, k))
	    not_p_xy = np.zeros((d, k))
	    for i in range(k):
	      # find counts of x given label y=i
	      x_i = X[y == i].sum(axis=0)
	      not_x_i = counts[i] - x_i
	      
	      # probability of x given label y=i with Laplace smoothing
	      p_xy[:, i] = (x_i + self.beta) / (counts[i] + self.beta * k)
	      not_p_xy[:, i] = (not_x_i + self.beta) / (counts[i] + self.beta * k)

	    self.p_y = p_y
	    self.p_xy = p_xy
	    self.not_p_xy = not_p_xy

