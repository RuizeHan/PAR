from spectral_cluster.spectralcluster import SpectralClusterer
from numpy import random

cluster = SpectralClusterer(
    min_clusters=1,
    max_clusters=12,
    autotune=None,
    laplacian_type=None,
    refinement_options=None)

relation_graph = random.random(size=(4, 4))
print(relation_graph)
social_group_predict = cluster.predict(relation_graph)
print(social_group_predict)