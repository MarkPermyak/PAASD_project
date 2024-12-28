import sys
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

def create_random_graph(nodes=500, connection_prob=0.5):
    graph = np.random.rand(nodes, nodes)
    graph = np.where(graph < connection_prob, np.inf, 1)
    np.fill_diagonal(graph, 0)

    return graph

def FloydWarshall(graph):
    dist = graph.copy()
    n = dist.shape[0]
    for k in range(n): 
        for i in range(n): 
            for j in range(n): 
                dist[i][j] = min(dist[i][j],dist[i][k]+ dist[k][j])
    return dist


def FloydWarshall_serial(i, dist, n, k):
    for j in range(n): 
        dist[i][j] = min(dist[i][j],dist[i][k]+ dist[k][j])
    return (i,dist[i])

def FloydWarshall_parallel(graph, processes=4):
    dist = graph.copy()
    n = dist.shape[0]
    pool = mp.Pool(processes=processes)
    for k in range(n):
        p = partial(FloydWarshall_serial, dist=dist,n=n,k=k)
        result_list = pool.map(p,range(n))
        for result in result_list:
            dist[result[0]] = result[1]
    pool.close()
    pool.join()
    return dist


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    np.random.seed(seed=123)
    graph = create_random_graph(nodes=N)
    print(f'Created graph with {N} nodes')

    start = time.time()
    dist_non_parallel = FloydWarshall(graph)
    end = time.time()

    time_non_parallel = end - start
    print(f'Non-parallel time: {time_non_parallel:.2f} seconds')


    processes_num = [2, 4, 6, 8, 10, 12]
    speedups = []
    
    for processes in processes_num:
        start = time.time()
        dist_parallel = FloydWarshall_parallel(graph, processes)
        end = time.time()

        time_parallel = end - start
        speedup = time_non_parallel / time_parallel
        speedups.append(speedup)

        solutions_equal = np.allclose(dist_non_parallel, dist_parallel, atol=1e-8)
        if solutions_equal:
            print(f'Non-parallel and parallel(processes={processes}) solutions are equal')
        else:
            print('Solutions are different :(')

        print(f'Parallel time with {processes} processes: {time_parallel:.2f} seconds')
        print(f'Faster by {speedup:.2f} times')


    plt.plot(processes_num, speedups, marker='o')
    plt.title('Speedups')
    plt.xlabel('Number of Processes')
    plt.ylabel('Non-parallel time / Parallel time')
    plt.xticks(processes_num)
    plt.grid()
    plt.tight_layout()
    plt.savefig('speedups.png')