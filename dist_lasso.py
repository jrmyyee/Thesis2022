import sys
import numpy as np
import scipy as sp
from numpy.linalg import inv
from scipy.sparse import csr_matrix as sparse

import matplotlib as mpl
from matplotlib import pyplot as plt

from pprint import pprint

from datetime import datetime

import networkx as nx
from network import genRobustNetwork as grn
from network import genMaliciousNodes

### UTIL FUNCTIONS
def chol(A):
	return np.linalg.cholesky(A)

def factor(A, rho):
	m = len(A)
	n = len(A[0])

	#If skinny
	if m >= n:
		'''
		Assuming speye means identity matrix
		'''
		L = chol(np.transpose(A).dot(A) + rho*np.identity(n))
	else:
		L = chol(np.identity(m) + 1/rho*(A.dot(np.transpose(A))))

	L = sparse(L).toarray()
	U = sparse(np.transpose(L)).toarray()

	return L, U

def shrinkage(x, kappa, n):
	return np.maximum(np.full((n,1), 0), x-np.full((n,1), kappa)) - np.maximum(np.full((n,1), 0), -x, np.full((n,1), -kappa))
	
def objective(A, b, lbda, x, z):
	obj = 0.5*np.sum(np.square(A.dot(x)-b)) + lbda*np.linalg.norm(z, 1)
	return obj

def moving_average(a, n=3):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

### UTIL FUNCTIONS END

def solve_lasso(A, b, lbda, rho, alpha, agents, neighbours, malicious=[], _shadowMalicious=False, p=None, set_iter = None, max_iter=1000, epsilon=1e-3, convergeAfter=10, switchNetwork=None):
	startTime = datetime.now()
	n = len(A[0])

	if _shadowMalicious:
		storedMalicious = malicious
		malicious = []

	xs = [np.array([np.zeros(n)]).T for i in range(agents)]
	us = [np.array([np.zeros(n)]).T for i in range(agents)]
	zs = [np.array([np.zeros(n)]).T for i in range(agents)]

	As = np.vsplit(A, agents)
	bs = np.vsplit(b, agents)

	m = len(As[0])

	Ls, Us = zip(*[factor(Ai, rho) for Ai in As])

	#Global objective calc
	g_history = []
	history = [[] for i in range(agents)]
	x_history = [[] for i in range(agents)]
	z_history = [[] for i in range(agents)]

	#For Boyd's convergence check
	old_z = None

	def convergence_check():
		if len(g_history) < convergeAfter:
			return False
		
		if len(g_history) >= convergeAfter*5:
			ma = moving_average(g_history, convergeAfter-1)
			grad = [ma[i]-ma[i-1] for i in range(1, len(ma))]
			if (np.array(grad) < epsilon).all():
				return True
		

		return False

	iteration = 0
	while iteration < max_iter:
		#Network switching (time varying graph)
		if switchNetwork is not None:
			if iteration % switchNetwork == 0 and iteration != 0:
				neighbours = nx.to_numpy_array(grn(agents, 4)) + np.identity(agents)

		#print(f"Iteration {iteration}")
		for i in range(agents):
			#Define malicious behaviour as noise
			if i in malicious:
				xs[i] = np.random.rand(n, 1)
				us[i] = np.random.rand(n, 1)
				continue

			old_x_hat = alpha*xs[i] + (1-alpha)*zs[i]

			#u update
			u = us[i] + (old_x_hat - zs[i])

			#x update
			q = np.transpose(As[i]).dot(bs[i]) + rho*(zs[i]-us[i])
			if m >= n:
				#If skinny
				x = np.linalg.lstsq(Us[i], np.linalg.lstsq(Ls[i], q)[0])[0]
			else:
				#If fat
				x = q/rho - (np.transpose(As[i]).dot(np.linalg.lstsq(Us[i], np.linalg.lstsq(Ls[i], As[i].dot(q), rcond=None)[0], rcond=None)[0]))/np.power(rho,2)

			#Update vals
			xs[i] = x
			us[i] = u

		#Calculate averages to transmit in communication
		xAvg = []
		uAvg = []
		for i in range(agents):
			'''
			vals -> matrix storing variable state across agents to be filtered
				- assumes 2d matrix, row indicates agent, column indicates feature/variable in vector
			i -> agent filtering is being done for
			neighbours -> neighbour matrix
			'''
			def com_filter(vals, i, neighbours):
				#Tentative values
				tent = []

				for index, cands in enumerate(np.transpose([vals[j] for j in range(len(vals)) if (neighbours[i][j]==1 and i!=j)])[0]):
					#Sort first
					cands.sort()

					#Filter
					#Value of given feature for agent i (add later)
					base = vals[i][index]

					#If less than p neighbour values greater than base
					if len([c for c in cands if c > base]) < p:
						#Remove all values larger than base
						cands = [c for c in cands if c <= base]
					else:
						#Remove p largest values
						cands = cands[:-p]

					#If less than p neighbour values smaller than base
					if len([c for c in cands if c < base]) < p:
						#Remove all values smaller than base
						cands = [c for c in cands if c >= base]
					else:
						#Remove p smallest values
						cands = cands[p:]

					#Add base to cands
					cands = np.append(cands, base)

					#Add average of cands to tent_x
					tent = np.append(tent, np.average(cands))

				#print(f"{xs[i]} vs {tent}")
				#print(tent)
				return tent

			def vec_filter(vals, i, neighbours, heuristic):
				tent = []
				base = heuristic(vals[i])

				'''
				Sort candidate vectors in order of their objective (ascending)
				'''
				cands = [(vals[j], heuristic(vals[j]), j) for j in range(len(vals)) if neighbours[i][j]==1 and i!=j]
				cands.sort(key=lambda x: x[1])

				#If less than p neighbour values greater than base
				if len([c for c in cands if c[1] > base]) < p:
					#Remove all values larger than base
					cands = [c for c in cands if c[1] <= base]
				else:
					#Remove p largest values
					cands = cands[:-p]

				#If less than p neighbour values smaller than base
				if len([c for c in cands if c[1] < base]) < p:
					#Remove all values smaller than base
					cands = [c for c in cands if c[1] >= base]
				else:
					#Remove p smallest values
					cands = cands[p:]

				#Add initial agent's values to cands
				#print(f"{i}: Accepting input from {[c[2] for c in cands]}")
				friendly = [c[2] for c in cands]
				friendly.append(i)

				cands = [c[0] for c in cands]

				cands.append(vals[i])
				
				tent = np.average(cands, axis=0)

				return tent, friendly

			#No filtering
			if p is None:
				xAvg.append(np.average([list(j) for j in zip(*[xs[k] for k in range(len(neighbours[i])) if neighbours[i][k] == 1])], axis=1))
				uAvg.append(np.average([list(j) for j in zip(*[us[k] for k in range(len(neighbours[i])) if neighbours[i][k] == 1])], axis=1))
			else:
				#COM filter
				#xAvg.append(np.array(com_filter(xs, i, neighbours)).reshape(n,1))
				#uAvg.append(np.array(com_filter(xs, i, neighbours)).reshape(n,1))

				xTent, xFriendly = vec_filter(xs, i, neighbours, h)
				uTent, uFriendly = vec_filter(us, i, neighbours, h)

				xAvg.append(np.array(xTent).reshape(n,1))
				uAvg.append(np.array(uTent).reshape(n,1))

		#Update u and x of each agent respectively
		for i in range(agents):
			xs[i] = xAvg[i]
			us[i] = uAvg[i]
			
		#z update
		for i in range(agents):
			x_hat = alpha*xs[i] + (1-alpha)*zs[i]
			z = shrinkage(x_hat + us[i], lbda/rho, n)
			zs[i] = z

			history[i].append(objective(As[i], bs[i], lbda, xs[i], zs[i]))
			x_history[i].append(xs[i])
			z_history[i].append(zs[i])
		
		#Calculate global objective using average from non-malicious agents
		if _shadowMalicious:
			avg_x = np.average([list(j) for j in zip(*[xs[k] for k in range(agents) if k not in storedMalicious])], axis=1)
			avg_z = np.average([list(j) for j in zip(*[zs[k] for k in range(agents) if k not in storedMalicious])], axis=1)
			#avg_u = np.average([list(j) for j in zip(*[zs[k] for k in range(len(neighbours[i])) if k not in storedMalicious])], axis=1)
		else:
			avg_x = np.average([list(j) for j in zip(*[xs[k] for k in range(agents) if k not in malicious])], axis=1)
			avg_z = np.average([list(j) for j in zip(*[zs[k] for k in range(agents) if k not in malicious])], axis=1)
			#avg_u = np.average([list(j) for j in zip(*[zs[k] for k in range(len(neighbours[i])) if k not in malicious])], axis=1)

		g_history.append(objective(A, b, lbda, avg_x, avg_z))

		#Boyd's convergence check
		if old_z is not None:
			r_norm = np.linalg.norm(avg_x - avg_z)
			s_norm = np.linalg.norm(-rho*(avg_z - old_z));

			ABSTOL = 1e-4
			RELTOL = 1e-2

			eps_pri = np.sqrt(n)*ABSTOL + RELTOL*np.maximum(np.linalg.norm(avg_x), np.linalg.norm(avg_z));
			eps_dual = np.sqrt(n)*ABSTOL + RELTOL*np.linalg.norm(rho*avg_u)

			if r_norm < eps_pri and s_norm < eps_dual:
				print(f"Solution converged at iteration {iteration}")
				break

		old_z = avg_z

		if convergence_check():
			print(f"Solution converged at iteration {iteration}")
			break

		iteration += 1

	print([k for k in range(agents) if k not in (storedMalicious if _shadowMalicious else malicious)])

	print(f"Time taken: {str(datetime.now() - startTime)}")
	return history, x_history, z_history, g_history
