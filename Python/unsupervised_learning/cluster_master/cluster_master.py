import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
rng = np.random.default_rng()
from scipy.cluster import  hierarchy
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import openml
from sklearn.model_selection import train_test_split
from genieclust.compare_partitions import normalized_clustering_accuracy
from genieclust import Genie
#from sklearn_extra.cluster import KMedoids
import optuna
import warnings

### CALINSKI-HARABASZ

def clusterize(model, X):
    labs = model.labels_
    clusters = []
    centers = []
    cluster_size = []
    # Wyznaczamy skupienia

    for l in np.unique(labs):
        cluster = X[labs == l]
        clusters.append(cluster)
        centers.append(np.array([np.mean(cluster[:,i]) for i in range(np.shape(cluster)[1])])) # liczymy centroidy skupień
        cluster_size.append(len(cluster))
    return (clusters, centers, cluster_size)

def sse(model, X):
    clusters, centers, _ = clusterize(model, X)
    n_clusters = len(centers)
    sse = 0
    for i in range(n_clusters):
        for point in clusters[i]:
            sse += np.sum((point - centers[i])**2)

    return sse

def calinski_harabasz(model, X):
    clusters, centers, _ = clusterize(model, X)
    centroid = np.array([np.mean(X[:,i]) for i in range(np.shape(X)[1])]) # Liczymy centroid całych danych
    n_clusters = len(centers)
    n = len(X)

    sce = np.sum([len(clusters[i]) * np.sum((centers[i]-centroid)**2) for i in range(n_clusters)]) # suma odchyłek skupień (sum of cluster error)

    return sce*(n-n_clusters)/((n_clusters-1)*sse(model, X))

### SYLWETKA

def average_distance(cluster, point):
    cluster_size = len(cluster)
    total_distance = 0
    total_distance = np.sum((cluster-point)**2)
    #print(f"Cluster size: {cluster_size}")
    #print(f"avg dist: {total_distance/(cluster_size-1)}")
    return total_distance/(cluster_size-1)

def min_outcluster_distance(model, X, cluster, point):
    clusters, centers, cluster_size = clusterize(model, X)
    distances = []

    # znajdujemy indeks wlasnego skupienia
    own_idx = None
    for i in range(len(centers)):
        if np.array_equal(clusters[i], cluster)==True:
            own_idx = i
            break
    #print(own_idx)
    for i in range(len(centers)):
        if i == own_idx:
            continue  # pomijamy własne skupienie

        distance = np.sum((clusters[i] - point) ** 2)
        #print(f"Distance: {distances}")
        distances.append(distance / cluster_size[i])
        #print(f"Distances: {distances}")
    #print(f"min dist: {np.min(distances)}")

    return np.min(distances)

def silhouette(model, X, cluster, point):
    if len(cluster) == 1: 
        return 0
    b = min_outcluster_distance(model, X, cluster, point)
    a = average_distance(cluster, point)
    m = np.max(np.array(b, a))
    sil = (b-a)/m
    #print(f"Silhouette: {sil}")

    return sil

def average_silhouette(model, X):
    return silhouette_score(X, model.labels_)
    clusters, centers, cluster_size = clusterize(model, X)
    average_silhouettes = []

    # sumujemy
    for c in clusters:
        s = 0
        for point in c:
            s += silhouette(model, X, c, point)
        average_silhouettes.append(s)
    
    # Wyliczamy srednia ogolna i po skupieniach
    total_silhouette = np.sum(average_silhouettes)/len(X)
    for i in range(len(average_silhouettes)):
        average_silhouettes[i] = average_silhouettes[i]/cluster_size[i]
    
    return total_silhouette#, average_silhouettes


def master_objective(trial, method, dataset, metric, param_dict):
    match method:
        case "kmeans":
            n_cluster = trial.suggest_int("n_clusters", param_dict["n_cluster_lower"], param_dict["n_cluster_upper"])
            model = KMeans(n_clusters=n_cluster, n_init = param_dict["n_init"])
        case "kmedoids":
            n_cluster = trial.suggest_int("n_clusters", param_dict["n_cluster_lower"], param_dict["n_cluster_upper"])
            model = KMedoids(n_clusters=n_cluster, init="k-medoids++")
        case "dbscan":
            eps = trial.suggest_float("eps", param_dict["eps_lower"], param_dict["eps_upper"])
            min_samples = trial.suggest_int("min_samples", param_dict["min_samples_lower"], param_dict["min_samples_upper"])
            model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        case "genie":
            n_cluster = trial.suggest_int("n_clusters", param_dict["n_cluster_lower"], param_dict["n_cluster_upper"])
            gini_threshold = trial.suggest_float("gini_threshold", 0, 1)
            model = Genie(n_clusters = n_cluster, gini_threshold = gini_threshold)
        case "hdbscan":
            min_cluster_size = trial.suggest_int("min_cluster_size", param_dict["min_cluster_size_lower"], param_dict["min_cluster_size_upper"])
            cluster_selection_epsilon = trial.suggest_float("cluster_selection_epsilon", param_dict["cluster_selection_epsilon_lower"], param_dict["cluster_selection_epsilon_upper"])
            max_cluster_size = trial.suggest_int("max_cluster_size", param_dict["max_cluster_size_lower"], param_dict["max_cluster_size_upper"])
            model = HDBSCAN(min_cluster_size = min_cluster_size, 
                            cluster_selection_epsilon = cluster_selection_epsilon,
                            max_cluster_size = max_cluster_size,
                            n_jobs = -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       
        model.fit(dataset)
    if len(np.unique(model.labels_)) == 1:
        return float("-Inf")
    return metric(model, dataset)

def get_params_dict():
    return {"n_cluster_lower": 2,
     "n_cluster_upper": 20,
     "n_init": 10,
     "eps_lower": 0.0,
     "eps_upper": 1.0,
     "min_samples_lower": 2,
     "min_samples_upper": 20,
     "min_cluster_size_lower": 2,
     "min_cluster_size_upper": 25,
     "max_cluster_size_lower": 50,
     "max_cluster_size_upper": 500,
     "cluster_selection_epsilon_lower": 0.0,
     "cluster_selection_epsilon_upper": 1,}

def cluster_master_study(X, method="kmeans", metric = "sse", n_trials = 100, param_dict = None):
    """
    cluster_master_study - łatwy sposób na optymalizację w analizie skupień.


    Parameters
    ----------

    X : Array-like
        Zbiór, na którym będzie przeprowadzana analiza skupień.

    method : string in {"kmeans", "kmedoids", "dbscan", "genie", "hdbscan", "optimize"}, default=kmeans
        Algorytm wyznaczania skupień.

    metric : string in {"sse", "calinski-harabasz", "silhouette"}, default=sse
        Metryka do optymalizacji.

    n_trials : int, default = 100
        Liczba iteracji algorytmu optymalizującego
    
    param_dict : dict, default = None
        Słownik ze wszystkimi parametrami potrzebnymi do optymalizacji. W przypadku podania None, zostanie wykorzystany słownik z funkcji get_params_dict().
        W przypadku podania własnego słownika nie trzeba zamieszczać w nim wszystkich zmiennych, wystarczą te używane w wybranej metodzie.
    """


    method_valid = {"kmeans", "kmedoids", "dbscan", "genie", "hdbscan", "optimize"}
    if method not in method_valid:
        raise ValueError("Niepoprawna nazwa modelu. Dozwolone nazwy: {kmeans, kmedoids, dbscan, genie, hdbscan, optimize}")
    
    metric_valid = {"sse", "calinski-harabasz", "silhouette"}
    if metric not in metric_valid:
        raise ValueError("Niepoprawna metryka. Dozwolone nazwy: {sse, calinski-harabasz, silhouette}")
    
    match metric:
        case "sse":
            metric_fun = sse
        case "calinski-harabasz":
            metric_fun = calinski_harabasz
        case "silhouette":
            metric_fun = average_silhouette

    if param_dict is None:
        param_dict = get_params_dict()

    study_master = optuna.create_study(direction = "maximize")
    study_master.optimize(lambda trial: master_objective(trial, method, X, metric_fun, param_dict), n_trials=n_trials)

    return study_master
    
def clusterize_labs(labs, X):
    clusters = []
    centers = []
    cluster_size = []
    # Wyznaczamy skupienia

    for l in np.unique(labs):
        cluster = X[labs == l]
        clusters.append(cluster)
        centers.append(np.array([np.mean(cluster[:,i]) for i in range(np.shape(cluster)[1])])) # liczymy centroidy skupień
        cluster_size.append(len(cluster))
    return (clusters, centers, cluster_size)

def sse_labs(labs, X):
    clusters, centers, _ = clusterize_labs(labs, X)
    n_clusters = len(centers)
    sse = 0
    for i in range(n_clusters):
        for point in clusters[i]:
            sse += np.sum((point - centers[i])**2)

    return sse



def calinski_harabasz_labs(labs, X):
    clusters, centers, _ = clusterize_labs(labs, X)
    centroid = np.array([np.mean(X[:,i]) for i in range(np.shape(X)[1])]) # Liczymy centroid całych danych
    n_clusters = len(centers)
    n = len(X)

    sce = np.sum([len(clusters[i]) * np.sum((centers[i]-centroid)**2) for i in range(n_clusters)]) # suma odchyłek skupień (sum of cluster error)

    return sce*(n-n_clusters)/((n_clusters-1)*sse_labs(labs, X))

def average_distance_labs(cluster, point):
    cluster_size = len(cluster)
    total_distance = 0
    total_distance = np.sum((cluster-point)**2)
    return total_distance/(cluster_size-1)

def min_outcluster_distance_labs(labs, X, cluster, point):
    clusters, centers, cluster_size = clusterize_labs(labs, X)
    distances = []

    # znajdujemy indeks wlasnego skupienia
    own_idx = None
    for i in range(len(centers)):
        if np.array_equal(clusters[i], cluster)==True:
            own_idx = i
            break
    #print(own_idx)
    for i in range(len(centers)):
        if i == own_idx:
            continue  # pomijamy własne skupienie

        distance = np.sum((clusters[i] - point) ** 2)
        #print(f"Distance: {distances}")
        distances.append(distance / cluster_size[i])
        #print(f"Distances: {distances}")
    #print(f"min dist: {np.min(distances)}")

    return np.min(distances)

def silhouette_labs(labs, X, cluster, point):
    if len(cluster) == 1:
        return 0
    b = min_outcluster_distance_labs(labs, X, cluster, point)
    a = average_distance_labs(cluster, point)
    m = np.max(np.array(b, a))
    sil = (b-a)/m
    #print(f"Silhouette: {sil}")

    return sil

def average_silhouette_labs(labs, X):
    return silhouette_score(X, labs)
    clusters, centers, cluster_size = clusterize_labs(labs, X)
    average_silhouettes = []

    # sumujemy
    for c in clusters:
        s = 0
        for point in c:
            s += silhouette_labs(labs, X, c, point)
        average_silhouettes.append(s)

    # Wyliczamy srednia ogolna i po skupieniach
    total_silhouette = np.sum(average_silhouettes)/len(X)
    for i in range(len(average_silhouettes)):
        average_silhouettes[i] = average_silhouettes[i]/cluster_size[i]

    return total_silhouette#, average_silhouettes