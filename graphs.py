# creates a function that generates a random graph with the verticies and the nodes
# and the edges
#
import random
def func(n, m):
    graph = {}
    for i in range(n):
        graph[i] = []
    for i in range(m):
        graph[random.randint(0, n-1)].append(random.randint(0, n-1))
    return graph


print(func(3,5))