import math
import pickle
import argparse
import numpy as np
import os
import time
import seaborn as sns
import tensorflow as tf
from human_dynamics.src.tf_smpl import batch_smpl
# from human_dynamics.src.tf_smpl.projection import batch_orth_proj_idrot
from functools import partial
from multiprocessing import Pool
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from matplotlib import animation
from tqdm import tqdm
from mg_utils import *

# container call to HMMR method
def get_smpl_3d_estimates(smpl, betas, thetas, b):
	n = thetas.shape[0]
	all_joints = []
	for i in tqdm(range(0, n, b), desc="Running SMPL"): # starting new session each time is oddly faster than keeping the same session and also uses less memory
		with tf.Session() as sess:
			beta_b = tf.Variable(betas[i:i+b].reshape(-1, 10), dtype=tf.float32)
			theta_b = tf.Variable(thetas[i:i+b].reshape(-1, 72), dtype=tf.float32)
			sess.run(tf.global_variables_initializer())
			joints = smpl(beta_b, theta_b).eval()
			all_joints.append(joints)
	# with tf.Session() as sess:
	# 	betas = tf.Variable(betas.reshape(-1, 10), dtype=tf.float32)
	# 	thetas = tf.Variable(thetas.reshape(-1, 72), dtype=tf.float32)
	# 	sess.run(tf.global_variables_initializer())
	# 	for i in tqdm(range(0, n, b), desc="Running SMPL"):
	# 		joints = smpl(betas[i:i+b], thetas[i:i+b]).eval()
	# 		all_joints.append(joints)
	return np.concatenate(all_joints)

# cams -> [scale, translation_x, translation_y]
# joints -> 3d coords (cartesian)
# kps -> 2d joint coords in camera perspective
# poses -> 3d rotation matrix per joint; global rotation of body + 23 relative joint rotations
# shapes -> SMPL PCA coefficients (ignore)
# verts -> triangular 3d mesh (ignore)
# omegas -> [shapes, poses_aa, cams] vector for mesh; can extract axis aligned joint rotations from inner 72 values
# J -> 3d joints of shaped mean human mesh
# J_transformed -> 3d joints of posed mesh
class Node():
	def __init__(self, cams, smpl_joints, node_index, thetas=None, betas=None, name="", index=-1, kps=None):
		# TODO: may have to opt to not interpolate on betas and instead use J somehow, in case betas dont have a linear relationship
		self.betas = betas
		self.cams = cams
		self.node_index = node_index

		self.name = name
		self.index = index
		self.joints = smpl_joints
		self.kps = kps

		self.quarts = aa2quart(thetas)
		self.root = smpl_joints[:, 0]

	def __repr__(self):
		return "<Node: " + self.name + "; frame " + str(self.index) + ">"

	def joints_3d(self):
		return self.joints

	def joints_2d(self):
		return self.kps.reshape(1, 25, 2)

	def node_type(self):
		return "frame"

class Transition(Node):
	def __init__(self, cams, coco_joints, node_index, A, B):
		super(Transition, self).__init__(cams, np.array([[0]]), node_index)
		self.A = A
		self.B = B
		self.joints = coco_joints

	def __repr__(self):
		return "<Transition: " + str(self.A) + " to " + str(self.B) + ">"

	def node_type(self):
		return "transition"

	def joints_2d(self):
		# joints (K, 25, 3)
		# cams (K, 3)
		return batch_orth_proj_idrot(self.joints, self.cams)

class SubjectDatabase():
	def __init__(self, root, output_dir, window_size, smpl_path):
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
				node = Node(data['cams'][i], data['J_transformed'][i], next_node, thetas=data['omegas'][i, 3:75], betas=data['shapes'][i], name=f.split('/')[-2], 
							index=i, kps=data['kps'][i])
				self.nodes.append(node)
				next_node += 1
			video_boundaries.append(next_node)
		
		self.window_size = window_size
		self.size = len(self.nodes)
		self.joint_dim = self.nodes[0].joints.shape[0]

		# initialize trivial edges and remove edges that stich the last frame of a video to the first frame of the next
		self.edges = np.zeros((self.size, self.size)) # M[i,j] = 1 means there exists a directed edge from node i to node j
		self.edges[np.arange(self.size-1), np.arange(1, self.size)] = 1
		video_boundaries = np.array(video_boundaries[:-1], dtype=np.int32)
		self.edges[video_boundaries-1, video_boundaries] = 0

		self.output_dir = output_dir
		self.smpl_path = smpl_path

		self.transitions = None

		print("DB init time: %f" % (time.time() - start))
		print("DB size: %d" % self.size)

	def sample_node(self):
		return self.nodes[np.random.randint(self.size)]

	def sample_edge(self):
		node_pairs = np.argwhere(self.edges==1)
		return node_pairs[np.random.randint(len(node_pairs))]

	def kovar_dist(self, tup, weights=None): # weights should be k repetitions of the same 24 values
		A, B = tup
		if weights is None:
			weights = np.ones(self.window_size * 24) #not hard coding 24 this apparently makes it run a lot slower lol
		if (B - self.window_size < 0) or (A+self.window_size >= self.size):
			return float("inf")
		assert(B-self.window_size >= 0)
		assert(A+self.window_size < self.size)
		window_A = np.concatenate([node.joints_3d() for node in self.nodes[A:A+self.window_size]]).T # [K, 24, 3] - > [3, 24K]; ordered (x, y, z) with y as vertical
		window_B =  np.concatenate([node.joints_3d() for node in self.nodes[B-self.window_size:B]]).T

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
		B_transformed = T@np.vstack((window_B,np.ones(self.window_size * 24)))
		assert(np.all(B_transformed[3]==1))
		return np.sum(weights * ((window_A - B_transformed[:3])**2))

	def save_graph(self):
		save(self.output_dir, 'edges.pkl', self.edges)

	def load_graph(self):
		filepath = os.path.join(self.output_dir, 'edges.pkl')
		self.edges = load(filepath)

	def store_transitions(self, transition_pairs, cams, coco_joints):
		N = len(transition_pairs)
		K = self.window_size
		self.transitions = [[None for _ in range(self.size)] for _ in range(self.size)] #T[i, j] stores the Transition Node that transitions between Node i and Node j
		for i in range(N):
			A, B = transition_pairs[i]
			self.transitions[A][B] = Transition(cams[K*i:K*(i+1)], coco_joints[K*i:K*(i+1)], self.size, A, B)
			self.size += 1

	def get_transition_pairs(self):
		all_edges = np.argwhere(self.edges == 1)
		return np.array([edge for edge in all_edges if (self.nodes[edge[0]].index != (self.nodes[edge[1]].index - 1))])

	def save_transitions(self, cams, coco_joints):
		save(self.output_dir, 'transitions.pkl', (cams, coco_joints))

	def load_transitions(self):
		filepath = os.path.join(self.output_dir, 'transitions.pkl')
		cams, coco_joints = load(filepath)
		transition_pairs = self.get_transition_pairs()
		self.store_transitions(transition_pairs, cams, coco_joints)

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

	def compute_interpolations(self, tup):
		A, B = tup
	
		nodes_A = self.nodes[A:A+self.window_size]
		nodes_B = self.nodes[B-self.window_size:B]

		vid_A = self.nodes[A].name
		vid_B = self.nodes[B].name

		transition_states = []
		name = "transition_" + vid_A + "_" + str(A) + "_to_" + vid_B + "_" + str(B)

		idxs = np.arange(self.window_size)
		cams = np.array([lerp(nodes_A[p].cams, nodes_B[p].cams, alpha(p, self.window_size)) for p in idxs])
		betas = np.array([lerp(nodes_A[p].betas, nodes_B[p].betas, alpha(p, self.window_size)) for p in idxs])
		thetas = np.array([quart2aa(slerp(nodes_A[p].quarts, nodes_B[p].quarts, alpha(p, self.window_size))) for p in idxs])
		return thetas, betas, cams

	def compute_transitions(self, smpl_batch_size):
		print('----------')
		print('Detecting transitions')
		print('----------')

		transition_pairs = self.get_transition_pairs()
		print('%d transitions detected' % len(transition_pairs))

		assert os.path.exists(self.smpl_path), "Ensure that human_dynamics is set up properly!"
		smpl = batch_smpl.SMPL(self.smpl_path)

		print('----------')
		print('Interpolating parameters')
		print('----------')
		start = time.time()
		thetas = []
		betas = []
		cams = []
		for pair in transition_pairs:
			t, b, c = self.compute_interpolations(pair)
			thetas.append(t)
			betas.append(b)
			cams.append(c)
		thetas = np.concatenate(thetas, axis=0).reshape(-1, 72) # (N *K, 96) -> (N*K, 96)
		betas = np.concatenate(betas, axis=0).reshape(-1, 10)
		cams = np.concatenate(cams, axis=0)
		# this kind of froze my computer
		# ---------------------------------------
		# with Pool() as p:
		# 	res = np.array(p.map(self.compute_interpolations, transition_pairs))
		# thetas = res[:, 0, :, :].reshape(-1, 72) # (N*K, 72)
		# betas = res[:, 1, :, :].reshape(-1, 10) # (N*K, 10)
		# cams = res[:, 2, :, :]
		time_taken = time.time() - start
		print("Interpolation compute time: %f" % time_taken)

		print('----------')
		print('Computing SMPL 3d estimations')
		print('----------')

		start = time.time()
		coco_joints = get_smpl_3d_estimates(smpl, betas, thetas, smpl_batch_size)
		time_taken = time.time() - start
		print("SMPL compute time: %f" % time_taken)

		print('----------')
		print('Storing transitions')
		print('----------')
		start = time.time()
		self.store_transitions(transition_pairs, cams, coco_joints)
		time_taken = time.time() - start
		print("Final storage time: %f" % time_taken)

		self.save_transitions(cams, coco_joints)

	def random_walk(self, vid_length_secs=10, fps=30):
		min_n_frames = video_length_secs * fps
		sequence_2d = []

		start_edge = self.sample_edge()
		sequence_2d.append(self.nodes[start_edge[0]].joints_2d())
		sequence_2d.append(self.nodes[start_edge[1]].joints_2d())
		curr_node = start_edge[1]

		n = 2
		while n < min_n_frames:
			next_node = np.random.choice(np.argwhere(self.edges[curr_node] == 1).flatten())
			if self.transitions[curr_node][next_node] is not None:
				sequence_2d.append(self.transitions[curr_node][next_node].joints_2d())
				n += self.window_size
			sequence_2d.append(self.nodes[next_node].joints_2d())
			n += 1
			curr_node = next_node
		return np.concatenate(sequence_2d, axis=0)

#----------------------------------- turn into args
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='data/dylan')
parser.add_argument('--output_dir', type=str, default="./graph")
parser.add_argument('--window_size', type=int, default=10)
parser.add_argument('--load_if_precomputed', action='store_true', default=False)
parser.add_argument('--smpl_batch_size', type=int, default=64)
parser.add_argument('--smpl_path', type=str, default='./human_dynamics/models/neutral_smpl_with_cocoplustoesankles_reg.pkl')
args = parser.parse_args()
#------------------------------------

db = SubjectDatabase(args.dataroot, args.output_dir, args.window_size, args.smpl_path)

if args.load_if_precomputed and os.path.exists(args.output_dir + '/edges.pkl'):
	print('----------')
	print('Loading precomputed graph')
	print('----------')
	db.load_graph()
else:
	db.compute_graph(threshold=2.5)

if args.load_if_precomputed and os.path.exists(args.output_dir + '/transitions.pkl'):
	print('----------')
	print('Loading precomputed transitions')
	print('----------')
	db.load_transitions()
else:
	db.compute_transitions(args.smpl_batch_size)
