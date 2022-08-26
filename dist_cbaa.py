import sys
import numpy as np
import scipy as sp
import cvxpy as cp
from matplotlib import pyplot as plt

from pprint import pprint

import networkx as nx
from network import genRobustNetwork as grn
from network import genMaliciousNodes

from scipy.optimize import linear_sum_assignment as lsa

'''
Case 1: 1 robot for 1 task
Case 2: more robots than tasks
'''

'''
Case 1 - handling multiple optimal is non-trivial -> assume solving for one unique solution
'''

def solve_cbaa(alphas, tasks, agents, rho, neighbours, malicious=[], p=0, max_iter=1000):
	n = agents
	m = tasks

	#For M matrix
	def M(row):
		temp = np.zeros((n,m))
		temp[row] = np.ones(m)
		return temp

	#X (each robot's understanding of each other's assignments)
	xs = [np.zeros((m, n)) for i in range(n)]

	#Dual variables
	qs = [np.zeros((m, n)) for i in range(n)]

	#Assign A, b, C, d based on case
	'''
	A, C matrices
	b, d vectors
	'''
	Im = np.identity(m)
	As = [None for i in range(n)]
	Cs = [None for i in range(n)]
	b = None
	d = None
	if n == m:
		#Set A and C (agent dependent)
		for i in range(n):
			As[i] = np.vstack((M(i), Im))
			Cs[i] = np.zeros((n, m))

		#Set b and d
		b = np.ones((1, n+m))
		d = np.zeros((1, m))
	elif n > m:
		for i in range(n):
			As[i] = M(i)
			Cs[i] = -1*Im
		b = np.ones((1, m))
		d = -1*np.ones((1, m))
	else:
		print("Invalid Case")
		return

	cost_hist = [[] for i in range(n)]

	iteration = 0
	trusted = neighbours
	while iteration < max_iter:
		#For DEBUG purposes
		if iteration % (max_iter//100) == 0:
			print(f"On iteration {iteration}")

		#Q update
		new_qs = []
		for i in range(n):
			#Xk_i - Xk_j
			total = np.zeros((m, n))
			for j in range(n):
				if j in trusted[i]:
					total += xs[i]-xs[j]
			new_qs.append(qs[i]+rho*total)
			
		#X update
		new_xs = []
		for i in range(n):
			if i in malicious:
				new_xs.append(np.random.rand(m, n))
				continue

			Qi = qs[i]

			x = cp.Variable((m, n))
			first = np.transpose(alphas[i, :])@x[i, :]
			second = np.ones((1, m))@cp.multiply(Qi, x)@np.ones((n, 1))
			third = rho*np.linalg.det(neighbours)*cp.square(cp.norm(x, "fro"))
			fsum = np.zeros((m, n))
			for j in range(n):
				if j in trusted[i]:
					fsum += cp.multiply((xs[i]+xs[j]), x)

			fourth = rho*np.ones((1,m))@fsum@np.ones((n,1))

			obj = cp.Minimize(first+second+third-fourth)

			constraints = [0 <= x, x <= 1]

			#Adding first sum constraint
			conSumOne = 0
			for j in range(n):	
				conSumOne = conSumOne + (As[j]@x[j, :])[None, :]
			constraints.append(conSumOne == b)

			#Adding second sum constraint
			conSumTwo = 0
			for j in range(n):	
				conSumTwo = conSumTwo + (Cs[j]@x[j, :])[None, :]
			constraints.append(conSumTwo <= d)

			prob = cp.Problem(obj, constraints)
			res = prob.solve()

			new_xs.append(x.value)

		#Update values
		qs = new_qs
		xs = new_xs

		#Communication/filtering
		def getTrustedNeighbours(vals, i, neighbours):
			trusted = []

			base = np.sum(np.multiply(alphas, xs[i]))
			#print(base)

			#Add all neighbours to trusted list first
			for j in range(agents):
				if neighbours[i][j] == 1 and i!=j:
					trusted.append((j, np.sum(np.multiply(alphas, xs[j]))))

			trusted = sorted(trusted, key=lambda x: x[1])

			if len([t for t in trusted if t[1] > base]) < p:
				#Remove all values larger than base
				trusted = [t for t in trusted if t[1] <= base]
			else:
				#Remove p largest values
				trusted = trusted[:-p]

			#If less than p neighbour values smaller than base
			if len([t for t in trusted if t[1] < base]) < p:
				#Remove all values smaller than base
				trusted = [t for t in trusted if t[1] >= base]
			else:
				#Remove p smallest values
				trusted = trusted[p:]

			return [t[0] for t in trusted]

		#Calculate trusted agents for next round
		if p != 0:
			trusted = [getTrustedNeighbours(xs, i, neighbours) for i in range(agents)]
		#pprint(trusted)

		#Calculate costs
		for i in range(n):
			cost = np.sum(np.multiply(alphas, np.round(xs[i])))
			cost_hist[i].append(cost)

		iteration += 1

	#Submit final output
	sols = [xs[i] for i in range(agents)]

	return cost_hist, sols
