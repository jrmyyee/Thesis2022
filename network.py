import random
import networkx as nx

from matplotlib import pyplot as plt

def genMaliciousNodes(agents, malicious):
	mlist = [i for i in range(agents)]

	#Remove (agents-malicious) to leave malicious agents left
	for i in range(agents-malicious):
		aoi = random.choice(mlist)
		mlist.remove(aoi)

	return mlist

'''
n-robust network for a system of x agents
'''
def genRobustNetwork(agents, n):
	net = None
	attempts = 0
	while attempts < 100:
		try:
			net = nx.random_regular_graph(n, agents)
			if not nx.is_connected(net):
				raise Exception
			break
		except Exception:
			net = None
			attempts += 1
	return net
