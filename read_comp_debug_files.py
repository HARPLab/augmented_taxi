import pstats
from pstats import SortKey
p = pstats.Stats('run_simulations_comp_cluster_random_stats_sample.txt')
p.strip_dirs().sort_stats().print_stats()