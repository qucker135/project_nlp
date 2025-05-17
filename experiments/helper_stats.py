import os


BGN = 'Response time: '

with open(os.path.join('reports', 'report_graph.txt'), 'r') as f:
    times = [float(line[len(BGN):]) for line in f.readlines() if line.startswith(BGN)]
    print(f"Average response time for graph: {sum(times) / len(times)}")
    print(f"Total response time for graph: {sum(times)}")
    print(times)
