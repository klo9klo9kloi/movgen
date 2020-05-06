import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
import os
import time
import seaborn as sns
import tensorflow as tf
from human_dynamics.src.tf_smpl.batch_smpl import SMPL
# from human_dynamics.src.tf_smpl.projection import batch_orth_proj_idrot
from functools import partial
from multiprocessing import Pool
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

smpl_path = './human_dynamics/models/neutral_smpl_with_cocoplustoesankles_reg.pkl'

def draw_pose(kps, H, W):
	"""
	0: Right heel
	1: Right knee
	2: Right hip
	3: Left hip
	4: Left knee
	5: Left heel
	6: Right wrist
	7: Right elbow
	8: Right shoulder
	9: Left shoulder
	10: Left elbow
	11: Left wrist
	12: Neck
	13: Head top
	14: nose
	15: left_eye
	16: right_eye
	17: left_ear
	18: right_ear
	19: left big toe
	20: right big toe
	21: Left small toe
	22: Right small toe
	23: L ankle
	24: R ankle
	"""
	body_part_pairs = [ 0, 24, 1, 2, 2, 8, 3, 9, 4, 3, 5,23, 6,7, 7,8, 8,12, 9,12, 10, 9, 11,10, 12,14, 14,13, 17,15,
            18,16, 19,23, 20,24, 21,23, 22,24, 23,4, 24,1, 2,3]
	kps = denormalize(kps)
	img = np.zeros((H, W, 3), dtype=np.uint8)
	body_colors = cm.rainbow(np.linspace(0, 1, len(body_part_pairs)))
	for i in range(0, len(body_part_pairs), 2):
		p1 = (int(kps[body_part_pairs[i]*2]), int(kps[body_part_pairs[i]*2 + 1]))
		p2 = (int(kps[body_part_pairs[i+1]*2]), int(kps[body_part_pairs[i+1]*2 + 1]))
		color = tuple(body_colors[i][:3] * 255)
		img = cv2.line(img, p1, p2, color, 5)
		img = cv2.circle(img, p2, 6, color, -1)
	return img

def denormalize(points, D=640):
    return (0.5 * ((points + 1.0) * D))

def slerp(p1, p2, t):
	"""Spherical Linear Interpolation
	"""
	assert(p1.shape == p2.shape)
	assert(p1.shape == (72,))
	p1 = p1.reshape(24, 3)
	p2 = p2.reshape(24, 3)
	p1 = (p1.T/np.sqrt(np.sum(p1**2, axis=1)))
	p2 = (p2.T/np.sqrt(np.sum(p2**2, axis=1)))
	dot_prods = np.clip(np.diag(np.dot(p1.T, p2)), 0.0, 1.0)
	omega = np.arccos(dot_prods)
	return ((np.sin((1-t)*omega)/np.clip(np.sin(omega), 1e-8, None) )*p1 + (np.sin(t*omega)/np.clip(np.sin(omega), 1e-8, None))*p2).T

def lerp(p1, p2, t):
	"""Normal Linear Interpolation
	"""
	assert(p1.shape == p2.shape)
	return t*p1 + (1-t)*p2

def alpha(p, k):
	"""Transition scheme used in Kovar et. al 2002
	"""
	if p <= -1:
		return 1
	if p > k:
		return 0
	return 2*((p+1)/k)**3 - 3*((p+1)/k)**2 + 1

# numpy re-implementation of the method from HMMR
def orth_proj_idrot(X, camera):
	camera = camera.reshape(-1, 1, 3)
	X_trans = X[:, :, :2] + camera[:, :, 1:]
	shape = X_trans.shape
	return (camera[:, :, 0] * X_trans.reshape(shape[0], -1)).reshape(shape)

# cams -> [scale, translation_x, translation_y]
# joints -> 3d coords (cartesian)
# kps -> 2d joint coords in camera perspective
# poses -> 3d rotation matrix per joint; global rotation of body + 23 relative joint rotations
# shapes -> SMPL PCA coefficients (ignore)
# verts -> triangular 3d mesh (ignore)
# omegas -> [shapes, poses_aa, cams] vector for mesh; can extract axis aligned joint rotations from inner 72 values

class Node():
	def __init__(self, cams, thetas, betas, node_index, name="", index=-1, joints=None, kps=None, pose_rot=None):
		assert(cams.shape[-1] == 3)
		assert(thetas.shape[-1] == 72)
		assert(betas.shape[-1] == 10)
		self.thetas = thetas
		self.betas = betas
		self.cams = cams
		self.node_index = node_index

		self.name = name
		self.index = index
		self.joints = joints 
		self.kps = kps
		self.pose_rot = pose_rot

	def __repr__(self):
		return "<Node: " + self.name + "; frame " + str(self.index) + ">"

	def cartesian_3d(self, axis_aligned=True):
		if self.joints is None:
			assert(os.path.exists(smpl_path), "Ensure that human_dynamics is set up properly!")
			if self.smpl is None:
				self.smpl = SMPL(smpl_path)
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				betas = tf.Variable(self.betas.reshape(1, -1), dtype=tf.float32)
				thetas = tf.Variable(self.thetas.reshape(1, -1), dtype=tf.float32)
				self.joints = smpl(betas, thetas)[0]
		return self.joints

	def cartesian_2d(self):
		if self.kps is None:
			pass #TODO: call projection function
		return self.kps

	# def polar_3d(self, axis_aligned=True):
	# 	if axis_aligned:
	# 		return self.pose_aa
	# 	else:
	# 		return self.pose_rot

class SubjectDatabase():
	def __init__(self, root, output_dir, window_size):
		start = time.time()
		pickle_files = []
		for (dirpath, subdirs, files) in os.walk(root):
			if 'hmmr_output.pkl' in files:
				pickle_files.append(os.path.join(dirpath, 'hmmr_output.pkl'))

		self.nodes = []
		next_node = 0
		video_boundaries = [] # store start of each new video
		for f in pickle_files:
			data = pickle.load(open(f, 'rb'))
			n_frames = len(data['kps'])
			for i in range(n_frames): #UNDO LATER
				node = Node(data['cams'][i], data['omegas'][i, 3:75], data['shapes'][i], next_node, name=f.split('/')[-2], index=i, 
							joints=data['joints'][i], kps=data['kps'][i], pose_rot=data['poses'][i])
				self.nodes.append(node)
				next_node += 1
			video_boundaries.append(next_node)
		
		self.window_size = window_size
		self.size = len(self.nodes)

		# initialize trivial edges and remove edges that stich the last frame of a video to the first frame of the next
		self.edges = np.zeros((self.size, self.size)) # M[i,j] = 1 means there exists a directed edge from node i to node j
		self.edges[np.arange(self.size-1), np.arange(1, self.size)] = 1
		video_boundaries = np.array(video_boundaries[:-1])
		self.edges[video_boundaries-1, video_boundaries] = 0

		self.output_dir = output_dir

		print("DB init time: %f" % (time.time() - start))
		print("DB size: %d" % self.size)

	def sample_node(self):
		return self.nodes[np.random.randint(self.size)]

	def kovar_dist(self, tup, weights=None): # weights should be k repetitions of the same 25 values
		A, B = tup
		if weights is None:
			weights = np.ones(self.window_size * 25)
		if (B - self.window_size < 0) or (A+self.window_size >= self.size):
			return float("inf")
		assert(B-self.window_size >= 0)
		assert(A+self.window_size < self.size)
		window_A = np.concatenate([node.cartesian_3d() for node in self.nodes[A:A+self.window_size]]).T # [K, 25, 3] - > [3, 25K]; ordered (x, y, z)
		window_B =  np.concatenate([node.cartesian_3d() for node in self.nodes[B-self.window_size:B]]).T

		inv_weight_sum = 1/np.sum(weights)
		A_mu = np.mean(weights*window_A, axis=1) # [3]
		B_mu = np.mean(weights*window_B, axis=1) 

		theta = np.arctan2(
			np.sum(weights * (window_A[0]*window_B[2] - window_B[0]*window_A[2])) - inv_weight_sum*(A_mu[0]*B_mu[2] - B_mu[0]*A_mu[2]), 
			np.sum(weights * (window_A[0]*window_B[0] + window_B[2]*window_A[2])) - inv_weight_sum*(A_mu[0]*B_mu[0] + B_mu[2]*A_mu[2]) 
			)
		x_0 = inv_weight_sum*(A_mu[0] - B_mu[0]*np.cos(theta)-B_mu[2]*np.sin(theta))
		z_0 = inv_weight_sum*(A_mu[2] + B_mu[0]*np.sin(theta) - B_mu[2]*np.cos(theta))

		T = np.array([[1, 0, 0, x_0],[0, 1, 0, 0],[0, 0, 1, z_0],[0, 0, 0, 1]]) @ np.array([[np.cos(theta), 0, np.sin(theta), 0],[0, 1, 0, 0],[-np.sin(theta), 0, np.cos(theta), 0],[0, 0, 0, 1]])
		B_transformed = T@np.vstack((window_B,np.ones(self.window_size * 25)))
		assert(np.all(B_transformed[3]==1))
		return np.sum(weights * ((window_A - B_transformed[:3])**2))

	def save_graph(self):
		assert(self.edges is not None)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		filepath = self.output_dir + '/edges.pkl'
		f = open(filepath, 'wb')
		pickle.dump(self.edges, f)

	def load_graph(self):
		assert(os.path.exists(output_dir))
		filepath = self.output_dir + '/edges.pkl'
		f = open(filepath, 'rb')
		self.edges = pickle.load(f)

	def compute_graph(self, weights=None, threshold=0.5, verbose=True):
		print('----------')
		print('Computing pairwise distances')
		print('----------')

		start = time.time()
		dist = self.kovar_dist((self.sample_node().index, self.sample_node().index), weights=weights)
		time_taken = time.time() - start
		
		print("Dist: %f" % dist)

		row, cols = np.unravel_index(np.arange((self.size**2)), (self.size, self.size))

		if verbose:
			print("Dist time: %f" %  time_taken)
			expected_time_seconds = len(row) * time_taken
			print("Expected dist calculation overhead: %d:%d:%d" % (expected_time_seconds//3600, expected_time_seconds%3600 // 60, expected_time_seconds%3600 % 60))

		coord_tuples = np.array(tuple(zip(row, cols)))

		start = time.time()
		with Pool() as p:
			dist = np.array(p.map(partial(self.kovar_dist, weights=weights.copy() if weights else None), coord_tuples))
		actual = time.time() - start

		if verbose:
			print("Actual dist calculation overhead: %d:%d:%d" % (actual//3600, actual%3600 // 60, actual%3600 % 60))
			print("Speedup ~ %f x" % (expected_time_seconds/actual))

		f = open('dist.pkl', 'wb')
		pickle.dump(dist, f)

		# N = int(dist.shape[0] ** (1/2))
		# sns.heatmap(dist_m.reshape(N, N))
		# plt.show()

		print('----------')
		print('Selecting potential edges')
		print('----------')
		valid_range = (dist != 0) & (dist != float("inf"))
		sns.distplot(dist[valid_range], bins=100)
		plt.show()

		candidate_transition_points, properties = find_peaks(-dist[valid_range], height=(float("-inf"), 0), distance=self.window_size)

		if verbose:
			print("Number of candidates: %d" % len(candidate_transition_points))

		filtered_transition_points = candidate_transition_points[dist[valid_range][candidate_transition_points] < threshold]
		edges = coord_tuples[valid_range][filtered_transition_points]

		if verbose:
			print("Filtered number of edges: %d" % len(edges))

		self.edges[edges[:, 0], edges[:, 1]] = 1

		print('----------')
		print('Pruning graph')
		print('----------')

		edges_crs = csr_matrix(self.edges)
		n_components, labels = connected_components(csgraph=edges_crs, directed=True, return_labels=True, connection='strong')
		print("Number of SCCs detected: %d" % n_components)
		unique, counts = np.unique(labels, return_counts=True)
		top_2_idx = np.argpartition(-counts, 2)[:2]
		top_2_scc_labels = unique[top_2_idx]
		print(top_2_scc_labels, counts[top_2_idx])

		# remove all edges (u, v) that don't connect two nodes in the largest SCC
		start = time.time()
		nodes = np.arange(self.size)
		u = np.zeros((self.size, self.size))
		v = np.zeros((self.size, self.size))
		u[nodes[labels == top_2_scc_labels[0]]] = 1
		v[:, nodes[labels == top_2_scc_labels[0]]] = 1
		self.edges[~np.logical_and(u, v)] = 0
		time_taken = time.time() - start
		print("Edge removal time: %f" % time_taken)
		print("Final num edges: %d" % self.edges.sum())

		# discard nodes without edges
		# TODO: store removed so can use during load graph
		start = time.time()
		num_out = self.edges.sum(axis=1)
		num_in = self.edges.sum(axis=0)
		num_nodes = self.size
		for i in range(self.size):
			if num_out[i] == 0 and num_in[i] == 0:
				self.nodes[i] == None
				num_nodes -= 1
		time_taken = time.time() - start
		print("Node removal time: %f" % time_taken)
		print("Final num nodes: %d" % num_nodes)

		print('----------')
		print('Saving final graph')
		print('----------')

		self.save_graph()

	def compute_transition(self, tup):
		A, B = tup

		nodes_A = self.nodes[A:A+self.window_size]
		nodes_B = self.nodes[B-self.window_size:B]

		vid_A = self.nodes[A].name
		vid_B = self.nodes[B].name

		transition_states = []
		for p in np.arange(self.window_size):
			name = "transition_" + vid_A + "_" + str(A) + "_to_" + vid_B + "_" + str(B)
			weight = alpha(p, self.window_size)
			cams = lerp(nodes_A[p].cams, nodes_B[p].cams, weight)
			betas = lerp(nodes_A[p].betas, nodes_B[p].cams, weight)
			thetas = slerp(nodes_A[p].thetas, nodes_B[p].thetas, weight)
			transition_states.append(Node(cams, thetas, betas, self.size, name=name, index=p))
			self.size += 1

		self.transitions[A][B] = transition_states

	def compute_transitions(self):
		print('----------')
		print('Computing transitions')
		print('----------')

		all_edges = np.argwhere(self.edges == 1)
		transition_pairs = np.array([edge for edge in all_edges if (self.nodes[edge[0]].index != (self.nodes[edge[1]].index - 1))])
		print('%d transitions detected' % len(transition_pairs))

		start = time.time()
		self.transitions = [[[] for _ in range(self.size)] for _ in range(self.size)] #T[i, j] stores the list of Nodes that transition between Node i and Node j
		with Pool() as p:
			p.map(self.compute_transition, transition_pairs)
		time_taken = time.time() - start
		print("Transition compute time: %f" % time_taken)

#----------------------------------- turn into args
output_dir = 'graph'
window_size = 10
load_if_precomputed = False 
#------------------------------------
db = SubjectDatabase('data/dylan', output_dir, window_size)

if load_if_precomputed and os.path.exists(output_dir + '/edges.pkl'):
	print('----------')
	print('Loading precomputed graph')
	print('----------')
	db.load_graph()
	db.compute_transitions()
else:
	db.compute_graph(threshold=2.5)
	db.compute_transitions()

	