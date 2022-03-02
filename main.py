import numpy as np
import matplotlib.pyplot as plt

# Constants
ZERO = 0
TOTAL_POINTS = 100
ONE = 1
NEGONE = -1
ONEHUNDRED = 100
CONVERGENCE_THRESHOLD = 1e-4
TWO = 2
T5 = 2.5
P4 = 0.4
P5 = 0.5
VARIANCE = [False, True]
BASES = [3,5,10,15]
LR = [0.01, 0.02]


def generate_inputs():
    x = np.random.uniform(low=0.0, high=1.0, size=(TOTAL_POINTS,)) # getting x values in [0.0, 1.0]
    x.sort(axis=0)
    noise = np.random.uniform(low=-0.1, high=0.1, size=(TOTAL_POINTS,)) # getting noise values in [-0.1, 0.1]
    y = (P5 + (P4*np.cos((x*np.pi*T5))))+noise # getting outputs
    y_actual = (P5 + (P4*np.cos((x*np.pi*T5))))
    return x, y, y_actual


def diff_var(X, k, closestCluster):
    clusters_lone = []
    variances = np.zeros(k) # keep track of variance of each cluster
    for i in range(k):
        points = X[closestCluster == i]  # points in cluster i
        if len(points) < TWO:  # keeping track of lone clusters
            clusters_lone.append(i)
        else:
            variances[i] = np.var(X[closestCluster == i])

    if len(clusters_lone) > ZERO:  # checking if we need to average variances of other clusters
        all_points = []
        for i in range(k):  # getting points that are not part of lone clusters
            if i not in clusters_lone:
                all_points.append(X[closestCluster == i])
        all_points = np.concatenate(all_points).ravel()
        variances[clusters_lone] = np.mean(
            np.var(all_points))  # setting variance of lone clusters to average of total variance
    return variances


def kmeans(X, k, same_var=False):
    clusters = np.random.choice(np.squeeze(X), size=k) # random cluster starting points
    old_clusters = clusters.copy() # keep track of old clusters

    done = False # for convergence

    while not done:
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :])) # distance of each point to each cluster
        closestCluster = np.argmin(distances, axis=ONE) # closest cluster for each point

        for i in range(k):
            points = X[closestCluster == i] # points in cluster i
            if len(points) > ZERO:
                clusters[i] = np.mean(points, axis=ZERO)

        done = np.linalg.norm(clusters - old_clusters) < CONVERGENCE_THRESHOLD # see if clusters have changed
        old_clusters = clusters.copy()

    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))  # distance of each point to each cluster
    closestCluster = np.argmin(distances, axis=ONE)  # closest cluster for each point

    if not same_var: # must calculate variance of each cluster
        variances = diff_var(X, k, closestCluster)
    else: # use dmax and assume same gaussian width
        dmax = max([np.abs(i-j) for i in clusters for j in clusters])
        var = (dmax / np.sqrt(TWO*k))**TWO
        variances = np.repeat(var, k)
    return clusters, variances


def gaussian_rbf(x, cluster_center, variance):
    return np.exp((NEGONE*((x-cluster_center)**TWO)) / (TWO*variance))


class RBF(object):
    def __init__(self, bases, learning_rate, same_var, epochs=ONEHUNDRED, gaus_rbf=gaussian_rbf):
        self.bases = bases
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.gaus_rbf = gaus_rbf
        self.same_var = same_var
        self.weights = np.random.randn(bases)
        self.bias = np.random.randn(ONE)

    def fit(self, X, y): # does all training
        self.clusters, self.variances = kmeans(X, self.bases, self.same_var)
        for i in range(self.epochs):
            for j in range(X.shape[ZERO]):
                # forward pass
                x_rbf = np.array([self.gaus_rbf(X[j], center, var) for center, var in zip(self.clusters, self.variances)])
                f = x_rbf.T.dot(self.weights) + self.bias


                loss = (y[j] - f).flatten()**TWO
                print('Epoch {} Loss: {}'.format(i+ONE, loss[ZERO]))

                # backward pass
                error = -(y[j] - f).flatten()
                print(error)

                # update
                self.weights = self.weights - (self.learning_rate*x_rbf*error)
                self.bias = self.bias - (self.learning_rate*error)

    def predictions(self, X): # predictions. Use after training using the 'fit' function.
        y_pred = []
        for j in range(X.shape[ZERO]):
            x_rbf = np.array([self.gaus_rbf(X[j], center, var) for center, var in zip(self.clusters, self.variances)])
            f = x_rbf.T.dot(self.weights) + self.bias
            y_pred.append(f)
        return np.array(y_pred)


def plot(X, y, y_pred, bases, learning_rate, same_var, y_act): # for plotting results
    tt = 'Different' if not same_var else 'Same'
    title_text = '{} Variance. Bases = {}. Learning Rate = {}'.format(tt, bases, learning_rate)
    plt.plot(X, y, '-o', label='Data with noise')
    plt.plot(X, y_act, '-o', label='Actual Function')
    plt.plot(X, y_pred, '-o', label='RBF')
    plt.legend()
    plt.title(title_text)
    plt.show()


def predict(X, y, bases, learning_rate, y_act, same_var=False): # prediction for specific parameter values
    rbf = RBF(bases, learning_rate, same_var=same_var)
    rbf.fit(X, y)
    y_pred = rbf.predictions(X)
    plot(X, y, y_pred, bases, learning_rate, same_var, y_act)


if __name__ == '__main__': # all tested parameter values
    x,y, y_actual = generate_inputs()
    for v in VARIANCE:
        for b in BASES:
            for lr in LR:
                predict(x, y, b, lr, y_actual, v)
