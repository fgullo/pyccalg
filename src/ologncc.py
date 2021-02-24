import os, sys, getopt, time
import pulp as plp

separator = '---------------'

def _running_time_ms(start):
	return int(round((time.time()-start)*1000))

def _load(dataset_path,random_edgeweight_generation):
	import random
	with open(dataset_path) as f:
		tot_min = 0
		id2vertex = {}
		vertex2id = {}
		edges = []
		graph = {}
		vertex_id = 0
		for line in f.readlines()[1:]:
			tokens = line.split()
			u = int(tokens[0])
			v = int(tokens[1])
			if random_edgeweight_generation:
				import random
				wp = random.uniform(random_edgeweight_generation[0],random_edgeweight_generation[1])
				wn = random.uniform(random_edgeweight_generation[0],random_edgeweight_generation[1])
			else:
				wp = float(tokens[2])
				wn = float(tokens[3])
			if wp != wn:
				if u not in vertex2id:
					vertex2id[u] = vertex_id
					id2vertex[vertex_id] = u
					vertex_id += 1
				if v not in vertex2id:
					vertex2id[v] = vertex_id
					id2vertex[vertex_id] = v
					vertex_id += 1
				uid = vertex2id[u]
				vid = vertex2id[v]
				if uid < vid:
					edges.append((uid,vid))
				else:
					edges.append((vid,uid))
				if uid not in graph.keys():
					graph[uid] = {}
				if vid not in graph.keys():
					graph[vid] = {}
				min_pn = min(wp,wn)
				tot_min += min_pn
				graph[uid][vid] = (wp-min_pn,wn-min_pn)
				graph[vid][uid] = (wp-min_pn,wn-min_pn)
		return (id2vertex,vertex2id,edges,graph,tot_min)

def _read_params():
	dataset_file = None
	random_edgeweight_generation = None
	solver = 'pulp'
	short_params = 'd:r:s:'
	long_params = ['dataset=','random=','solver=']
	try:
		arguments, values = getopt.getopt(sys.argv[1:], short_params, long_params)
	except getopt.error as err:
		print('ologncc.py -d <dataset_file> [-r <rnd_edge_weight_LB,rnd_edge_weight_UB>] [-s <solver>]')
		sys.exit(2)
	for arg, value in arguments:
		if arg in ('-d', '--dataset'):
			dataset_file = value
		elif arg in ('-r', '--random'):
			random_edgeweight_generation = [float(x) for x in value.split(',')]
		elif arg in ('-s', '--solver'):
			solver = value.lower()
	return (dataset_file,random_edgeweight_generation,solver)

def _map_cluster(cluster,id2vertex):
	return {id2vertex[u] for u in cluster}

def _vertex_pair_id(i,j,n):
	lb = min(i,j)
	ub = max(i,j)
	return int(lb*n-(lb*(lb+1)/2))+ub-lb-1

def _vertex_pair_ids(n):
	id2vertexpair = {}
	id = 0
	for i in range(n-1):
		for j in range(i+1,n):
			id2vertexpair[id] = (i,j)
			id += 1
	return id2vertexpair

def _linear_program_scipy(num_vertices,edges,graph):
	vertex_pairs = int(num_vertices*(num_vertices-1)/2)
	A = []
	for i in range(num_vertices-2):
		for j in range(i+1,num_vertices-1):
			for k in range(j+1,num_vertices):
				#xij + xjk >= xik  <=>  xik - xij - xjk <= 0
				a = [0]*vertex_pairs
				ik = _vertex_pair_id(i,k,num_vertices)
				ij = _vertex_pair_id(i,j,num_vertices)
				jk = _vertex_pair_id(j,k,num_vertices)
				a[ik] = 1
				a[ij] = -1
				a[jk] = -1
				A.append(a)
	b = [0]*len(A)
	c = [0]*vertex_pairs
	for (u,v) in edges:
		uv = _vertex_pair_id(u,v,num_vertices)
		(wp,wn) = graph[u][v]
		if wp != wn:
			if wp < wn: #(u,v) \in E^-
				c[uv] = -(wn-wp)
			else: #(u,v) \in E^+
				c[uv] = (wp-wn)
	return (A,b,c)

def _solve_lp_scipy(A,b,c):
	from scipy.optimize import linprog
	lp_solution = linprog(c, A_ub=A, b_ub=b, bounds=[(0,1)])
	lp_var_assignment = lp_solution['x']
	for i in range(len(lp_var_assignment)):
		if lp_var_assignment[i] < 0:
			lp_var_assignment[i] = 0
		elif lp_var_assignment[i] > 1:
			lp_var_assignment[i] = 1
	return lp_var_assignment

def _linear_program_pulp(num_vertices,edges,graph):
	#see https://medium.com/opex-analytics/optimization-modeling-in-python-pulp-gurobi-and-cplex-83a62129807a
	opt_model = plp.LpProblem(name='GeneralWeightedCC')
	vertex_pairs = int(num_vertices*(num_vertices-1)/2)

	x_vars  = {i: plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=10, name='x_{0}'.format(i)) for i in range(vertex_pairs)}

	c_count = 0
	constraints = {}
	for i in range(num_vertices-2):
		for j in range(i+1,num_vertices-1):
			for k in range(j+1,num_vertices):
				#xij + xjk >= xik  <=>  xik - xij - xjk <= 0
				ik = _vertex_pair_id(i,k,num_vertices)
				ij = _vertex_pair_id(i,j,num_vertices)
				jk = _vertex_pair_id(j,k,num_vertices)
				expr = plp.LpAffineExpression([(x_vars[ik],1), (x_vars[ij],-1), (x_vars[jk],-1)])
				constraints[c_count] = opt_model.addConstraint(plp.LpConstraint(e=expr,sense=plp.LpConstraintLE,rhs=0,name='constraint_{0}'.format(c_count)))
				c_count += 1

	obj_expr = []
	for (u,v) in edges:
		uv = _vertex_pair_id(u,v,num_vertices)
		(wp,wn) = graph[u][v]
		if wp != wn:
			if wp < wn: #(u,v) \in E^-
				obj_expr.append(-(wn-wp)*x_vars[uv])
			else: #(u,v) \in E^+
				obj_expr.append((wn-wp)*x_vars[uv])
	objective = plp.lpSum(obj_expr)

	opt_model.sense = plp.LpMinimize
	opt_model.setObjective(objective)

	return opt_model

def _solve_lp_pulp(model):
	model.solve()
	lp_var_assignment = [x.varValue for x in model.variables()]
	return lp_var_assignment

def _sorted_distances(u,valid_vertices,num_vertices,x):
	du = []
	for v in valid_vertices:
		if v != u:
			du.append((v,x[_vertex_pair_id(u,v,num_vertices)]))
	return sorted(du,key=lambda f: f[1])

def _cut(ball,valid_vertices,graph):
	cut = 0
	for u in ball:
		if u in graph:
			for v in graph[u]:
				if v in valid_vertices and v not in ball:
					cut += graph[u][v][0]
	return cut

def _incremental_cut(old_ball,new_ball_vertices,valid_vertices,graph):
	incr_cut = 0
	for u in new_ball_vertices:
		if u in graph:
			for v in graph[u]:
				if v in valid_vertices:
					if v in old_ball:
						incr_cut -= graph[u][v][0]
					elif v not in new_ball_vertices:
						incr_cut += graph[u][v][0]
	return incr_cut

def _vol(ball,valid_vertices,graph,num_vertices,x,r):
	vol = 0
	for u in ball:
		if u in graph:
			for v in graph[u]:
				if v in valid_vertices:
					xuv = x[_vertex_pair_id(u,v,num_vertices)]
					if v in ball:
						if u<v:#check not to consider xuv two times
							vol += xuv*graph[u][v][0]
					else:
						vol += (r-xuv)*graph[u][v][0]
	return vol

def _vol_whole_graph(graph,num_vertices,x):
	vol = 0
	for u in graph.keys():
		for v in graph[u]:
			if u < v:#check not to consider xuv two times
				xuv = x[_vertex_pair_id(u,v,num_vertices)]
				cuv = graph[u][v][0]
				vol += xuv*cuv
	return vol

def _round(x,id2vertexpair,id2vertex,edges,graph,const):
	import random
	from math import log
	clusters = []
	n = len(id2vertex)
	remaining_vertices = set(id2vertex.keys())
	shuffled_vertices = list(id2vertex.keys())
	random.shuffle(shuffled_vertices)
	F = _vol_whole_graph(graph,n,x)

	for u in shuffled_vertices:
		if u in remaining_vertices:
			du = _sorted_distances(u,remaining_vertices,n,x)
			ball = {u}
			cut = _cut(ball,remaining_vertices,graph)
			vol = F/n #default initial volume of a ball
			while cut > const*log(n+1)*vol and du:#'log' returns natural logarithm
				r = du[0][1]#minimum distance in du
				i = 0
				while i<len(du) and du[i][1]<=r:
					i += 1
				new_ball_vertices = {v for (v,d) in du[0:i]}
				incr_cut = _incremental_cut(ball,new_ball_vertices,remaining_vertices,graph)
				cut += incr_cut
				ball.update(new_ball_vertices)
				vol = _vol(ball,remaining_vertices,graph,n,x,r)#cannot be done incrementally as r changes in every iteration, so all vertices in current ball must be visited again
				"""
				#####################
				#DEBUG
				cut_check = _cut(ball,remaining_vertices,graph)
				#cut = cut_check
				tolerance = 0.0000000001
				if abs(cut-cut_check) > tolerance:
					print('ERROR! cut incremental: %s, cut from scratch: %s' %(str(cut),str(cut_check)))
					sys.exit()
				#####################
				"""
				du = du[i:]
			clusters.append(ball)
			for v in ball:
				remaining_vertices.remove(v)
	return clusters

def _kwikcluster(id2vertex,graph):
	import random
	clusters = []
	n = len(id2vertex)
	remaining_vertices = set(id2vertex.keys())
	shuffled_vertices = list(id2vertex.keys())
	random.shuffle(shuffled_vertices)

	for u in shuffled_vertices:
		if u in remaining_vertices:
			cluster = {u}
			if u in graph:
				for v in graph[u]:
					if v in remaining_vertices:
						(wp,wn) = graph[u][v]
						if wp > wn:
							cluster.add(v)
			clusters.append(cluster)
			for v in cluster:
				remaining_vertices.remove(v)

	return clusters

def _CC_cost(clustering,graph):
	cost = 0

	vertex2cluster = {}
	cid = 0
	for cluster in clustering:
		for u in cluster:
			vertex2cluster[u] = cid
		cid += 1

	for u in graph.keys():
		for v in graph[u]:
			(wp,wn) = graph[u][v]
			if vertex2cluster[u] == vertex2cluster[v]:
				cost += wn
			else:
				cost += wp

	return cost

def _all_edgeweights_sum(graph):
	sum = 0
	for u in graph.keys():
		for v in graph[u]:
			if u<v:
				(wp,wn) = graph[u][v]
				sum += wp
				sum += wn
	return sum

def _max_edgeweight_gap(graph):
	maxgap = 0
	for u in graph.keys():
		for v in graph[u]:
			if u<v:
				(wp,wn) = graph[u][v]
				maxgap = max(maxgap,abs(wp-wn))
	return maxgap

if __name__ == '__main__':
	#read parameters
	(dataset_file,random_edgeweight_generation,solver) = _read_params()

	#load dataset
	print(separator)
	print('Loading dataset \'%s\'...' %(dataset_file))
	start = time.time()
	(id2vertex,vertex2id,edges,graph,tot_min) = _load(dataset_file,random_edgeweight_generation)
	runtime = _running_time_ms(start)
	n = len(id2vertex)
	m = len(edges)
	vertex_pairs = n*(n-1)/2
	vertex_triples = n*(n-1)*(n-2)/6
	print('Dataset successfully loaded in %d ms' %(runtime))
	print('#vertices: %d' %(n))
	print('#edges: %d' %(m))
	print('#vertex pairs: %d' %(vertex_pairs))
	print('#vertex triples: %d' %(vertex_triples))
	if random_edgeweight_generation:
		print('Edge weights randomly generated from [%s,%s]' %(random_edgeweight_generation[0],random_edgeweight_generation[1]))
	all_edgeweights_sum = _all_edgeweights_sum(graph)
	max_edgeweight_gap = _max_edgeweight_gap(graph)
	print('Global condition (without tot_min): %f >= %f ?' %(all_edgeweights_sum/vertex_pairs,max_edgeweight_gap))
	print('Global condition (including tot_min): %f >= %f ?' %((all_edgeweights_sum+tot_min)/vertex_pairs,max_edgeweight_gap))
	print('Solver: %s' %(solver))

	#baseline CC costs
	print(separator)
	singlecluster_cost = _CC_cost([set(id2vertex.keys())],graph) + tot_min
	allsingletons_cost = _CC_cost([{u} for u in id2vertex.keys()],graph) + tot_min
	print('CC cost of \'whole graph in one cluster\' solution: %f (tot_min: %f)' %(singlecluster_cost,tot_min))
	print('CC cost of \'all singletons\' solution: %f (tot_min: %f)' %(allsingletons_cost,tot_min))

	#run KwikCluster algorithm (to have some baseline results)
	print(separator)
	print('Running KwikCluster algorithm...')
	start = time.time()
	kc_clustering = _kwikcluster(id2vertex,graph)
	runtime = _running_time_ms(start)
	print('KwikCluster algorithm successfully executed in %d ms' %(runtime))
	kc_cost = _CC_cost(kc_clustering,graph) + tot_min
	print('CC cost of KwikCluster\'s output clustering: %f (tot_min: %f)' %(kc_cost,tot_min))
	print('KwikCluster\'s output clustering:')
	c = 1
	for cluster in kc_clustering:
		mapped_cluster = _map_cluster(cluster,id2vertex)
		print('Cluster ' + str(c) + ': ' + str(sorted(mapped_cluster)))
		c += 1

	#build linear program
	print(separator)
	print('O(log n)-approximation algorithm - Building linear program (solver: %s)...' %(solver))
	start = time.time()
	id2vertexpair = _vertex_pair_ids(n)
	model = None
	A = None
	b = None
	c = None
	c_nonzero = None
	if solver == 'pulp':
		model = _linear_program_pulp(n,edges,graph)
	elif solver == 'scipy':
		(A,b,c) = _linear_program_scipy(n,edges,graph)
		c_nonzero = len([x for x in c if x != 0])
	else:
		raise Exception('Solver \'%s\' not supported' %(solver))
	runtime = _running_time_ms(start)
	print('Linear program successfully built in %d ms' %(runtime))
	if solver == 'scipy':
		print('#variables: %d (must be equal to #vertex pairs)' %(len(c)))
		print('#inequality constraints: %d (must be equal to #vertex triples)' %(len(A)))
		print('#non-zero entries in cost vector: %d (must be <= #edges)' %(c_nonzero))

	#solving linear program
	print(separator)
	print('O(log n)-approximation algorithm - Solving linear program (solver: %s)...' %(solver))
	start=time.time()
	lp_var_assignment = None
	if solver == 'pulp':
		lp_var_assignment = _solve_lp_pulp(model)
	elif solver == 'scipy':
		lp_var_assignment = _solve_lp_scipy(A,b,c)
	else:
		raise Exception('Solver \'%s\' not supported' %(solver))
	runtime = _running_time_ms(start)
	#########
	#DEBUG
	print(lp_var_assignment)
	#########
	print('Linear program successfully solved in %d ms' %(runtime))
	print('size of the solution array: %d (must be equal to #variables)' %(len(lp_var_assignment)))

	#rounding lp solution
	print(separator)
	print('O(log n)-approximation algorithm - Rounding the LP solution...')
	eps = 0.0000000001
	start=time.time()
	clustering = _round(lp_var_assignment,id2vertexpair,id2vertex,edges,graph,2+eps)
	runtime = _running_time_ms(start)
	print('LP-rounding successfully performed in %d ms' %(runtime))
	cost = _CC_cost(clustering,graph) + tot_min
	print('CC cost of O(log n)-approximation algorithm\'s output clustering: %f (tot_min: %f)' %(cost,tot_min))
	print('O(log n)-approximation algorithm\'s output clustering:')
	c = 1
	for cluster in clustering:
		mapped_cluster = _map_cluster(cluster,id2vertex)
		print('Cluster ' + str(c) + ': ' + str(sorted(mapped_cluster)))
		c += 1
