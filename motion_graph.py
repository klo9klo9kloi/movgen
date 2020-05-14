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
from viz import animate_sequence
import matplotlib.pyplot as plt
import sys

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
		# self.root = smpl_joints[:, 0]

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
		return orth_proj_idrot(self.joints, self.cams)

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
		self.video_boundaries = video_boundaries

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

		# make sure A and B aren't close to any boundaries
		if (B - self.window_size < 0) or np.sum(np.abs(B - self.video_boundaries) < self.window_size) > 0:
			return 9999

		if (A+self.window_size >= self.size) or np.sum(np.abs(self.video_boundaries-A) < self.window_size) > 0:
			return 9999

		if (np.abs(A - B) < self.window_size):
			return 9999

		window_A = np.concatenate([node.joints_3d() for node in self.nodes[A:A+self.window_size]]).T # [K, 24, 3] - > [3, 24K]; ordered (x, y, z) with y as vertical
		window_B =  np.concatenate([node.joints_3d() for node in self.nodes[B-self.window_size+1:B+1]]).T

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
		frame_plus_one = (all_edges[:, 0]+1 == all_edges[:, 1]) #don't include natural edges
		edges_needing_transition = all_edges[~frame_plus_one]
		return np.array([edge for edge in edges_needing_transition])

	def get_edge_filter(self, edge_pairs):
		backward_edges = (edge_pairs[:, 0] > edge_pairs[:, 1])
		cross_clip_edges = np.array([(self.nodes[edge[0]].name != self.nodes[edge[1]].name) for edge in edge_pairs])
		return (backward_edges | cross_clip_edges)

	def save_transitions(self, cams, coco_joints):
		save(self.output_dir, 'transitions.pkl', (cams, coco_joints))

	def load_transitions(self):
		filepath = os.path.join(self.output_dir, 'transitions.pkl')
		cams, coco_joints = load(filepath)
		transition_pairs = self.get_transition_pairs()
		self.store_transitions(transition_pairs, cams, coco_joints)

	def compute_graph(self, weights=None, threshold=0.5, verbose=True):
		# print('----------')
		# print('Computing pairwise distances')
		# print('----------')

		# start = time.time()
		# dist = self.kovar_dist((self.sample_node().index, self.sample_node().index), weights=weights)
		# time_taken = time.time() - start
		
		# print("Dist: %f" % dist)

		row, cols = np.unravel_index(np.arange((self.size**2)), (self.size, self.size))
		coord_tuples = np.array(tuple(zip(row, cols)))

		# if verbose:
		# 	print("Dist time: %f" %  time_taken)
		# 	expected_time_seconds = len(row) * time_taken
		# 	print("Expected dist calculation overhead: %d:%d:%d" % (expected_time_seconds//3600, expected_time_seconds%3600 // 60, expected_time_seconds%3600 % 60))

		# start = time.time()
		# with Pool() as p:
		# 	dist = np.array(p.map(partial(self.kovar_dist, weights=weights.copy() if weights else None), coord_tuples))
		# actual = time.time() - start

		# if verbose:
		# 	print("Actual dist calculation overhead: %d:%d:%d" % (actual//3600, actual%3600 // 60, actual%3600 % 60))
		# 	print("Speedup ~ %f x" % (expected_time_seconds/actual))

		# f = open('dist.pkl', 'wb')
		# pickle.dump(dist, f)

		f = open('dist_10.pkl' ,'rb')
		dist = pickle.load(f)
		# dist[dist==float("inf")] = 9999 #REMOVE


		print('----------')
		print('Selecting potential edges')
		print('----------')
		# sns.distplot(dist, bins=100)
		# plt.show()

		candidate_transition_points, properties = find_peaks(-dist, distance=30) #this can probably be a hyperparameter
		print(dist[candidate_transition_points].max())
		print(dist[candidate_transition_points].min())

		if verbose:
			print("Number of candidates: %d" % len(candidate_transition_points))

		edge_filter = self.get_edge_filter(coord_tuples)
		candidates = np.zeros(len(dist)).astype(bool)
		candidates[candidate_transition_points] = True
		below_threshold = (dist < threshold)
		edges = coord_tuples[candidates & below_threshold & edge_filter]

		if verbose:
			print("Filtered number of transitions: %d" % len(edges))

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
		print("#1 SCC %d, count %d" % (top_2_scc_labels[0], counts[top_2_idx[0]]))
		print("#2 SCC %d, count %d" % (top_2_scc_labels[1], counts[top_2_idx[1]]))

		# remove all edges (u, v) that don't connect two nodes in the largest SCC
		start = time.time()
		nodes = np.arange(self.size)
		u = np.zeros((self.size, self.size))
		v = np.zeros((self.size, self.size))
		u[nodes[labels == top_2_scc_labels[0]]] = 1
		# u[nodes[labels == top_2_scc_labels[1]]] = 1
		v[:, nodes[labels == top_2_scc_labels[0]]] = 1
		# v[:, nodes[labels == top_2_scc_labels[0]]] = 1
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

		node_A = self.nodes[A]
		node_B = self.nodes[B]

		# especially with a small dataset, 
		# the motions leading to/following after
		# the match very rarely make sense
		# to interpolate between

		# JK, the way kovar dist is defined
		# this shouldn't be a problem
		# so either interpolation is wrong or 
		# kovar dist imeplementation is wrong
		#-------------------------------------
		nodes_A = self.nodes[A:A+self.window_size]
		nodes_B = self.nodes[B-self.window_size+1:B+1]

		vid_A = self.nodes[A].name
		vid_B = self.nodes[B].name

		transition_states = []
		name = "transition_" + vid_A + "_" + str(A) + "_to_" + vid_B + "_" + str(B)

		idxs = np.arange(0, self.window_size, 1)
		cams = np.array([lerp(node_A.cams, node_B.cams, alpha(i, self.window_size)) for i in idxs])
		betas = np.array([lerp(node_A.betas, node_B.betas, alpha(i, self.window_size)) for i in idxs])
		# input weight here  should represent how much we've moved from A already; so have to 1-alpha
		thetas = np.array([quart2aa(slerp(nodes_A[i].quarts, nodes_B[i].quarts, 1-alpha(i, self.window_size))) for i in idxs]) 
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

	def random_walk(self, vid_length_secs=10, fps=30, method='pseudo'):
		min_n_frames = vid_length_secs * fps
		sequence_2d = []

		start_edge = self.sample_edge()
		sequence_2d.append(self.nodes[start_edge[0]].joints_2d())
		sequence_2d.append(self.nodes[start_edge[1]].joints_2d())
		curr_node = start_edge[1]

		transition_method = self.get_transition_method(method)

		n = 2
		while n < min_n_frames:
			next_node = transition_method(np.argwhere(self.edges[curr_node] == 1).flatten(), curr_node)
			if self.transitions[curr_node][next_node] is not None:
				print(curr_node, next_node)
				print("Transitioning from %s to %s" % (self.nodes[curr_node].name, self.nodes[next_node].name) )
				# fig, ax = plt.subplots(2)
				# ax[0].imshow(draw_pose(self.nodes[curr_node].joints_2d().flatten(), 640, 640))
				# ax[1].imshow(draw_pose(self.nodes[next_node].joints_2d().flatten(), 640, 640))
				# plt.show()
				a = self.transitions[curr_node][next_node].joints_2d()
				sequence_2d.append(a)
				n += self.window_size
			sequence_2d.append(self.nodes[next_node].joints_2d())
			n += 1
			curr_node = next_node
		print()
		return np.concatenate(sequence_2d, axis=0)

	def creative_walk(self, vid_length_secs=10, fps=30):
		min_n_frames = vid_length_secs * fps
		sequence_2d = []

		start_edge = self.sample_edge()
		sequence_2d.append(self.nodes[start_edge[0]].joints_2d())
		sequence_2d.append(self.nodes[start_edge[1]].joints_2d())
		curr_node = start_edge[1]

		edge_weights = self.edges.copy() #probability of taking a specific edge
		last_cross = -999
		last_transition = -999

		n = 2
		while n < min_n_frames:
			choices = np.argwhere(self.edges[curr_node] == 1).flatten()
			potential_cross = np.array([(self.nodes[node].name != self.nodes[curr_node].name) for node in choices])

			if np.sum(potential_cross) > 0 and n-last_cross > 60: # prioritize a clip cross if the last time we did was longer than two seconds ago
				next_node = np.random.choice(choices[potential_cross])
				last_cross = n
			else: #otherwise, take random back and forward edges within the same clip with weighted probability
				weights = edge_weights[curr_node][choices]/np.sum(edge_weights[curr_node])
				next_node = np.random.choice(choices, p=weights)

			if self.transitions[curr_node][next_node] is not None:
				print(curr_node, next_node)
				print("Transitioning from %s to %s" % (self.nodes[curr_node].name, self.nodes[next_node].name) )
				# fig, ax = plt.subplots(2)
				# ax[0].imshow(draw_pose(self.nodes[curr_node].joints_2d().flatten(), 640, 640))
				# ax[1].imshow(draw_pose(self.nodes[next_node].joints_2d().flatten(), 640, 640))
				# plt.show()
				a = self.transitions[curr_node][next_node].joints_2d()
				sequence_2d.append(a)
				n += len(a)
			sequence_2d.append(self.nodes[next_node].joints_2d())
			n += 1
			curr_node = next_node
		print()
		return np.concatenate(sequence_2d, axis=0)

	def get_transition_method(self, method_name):
		if method_name == 'pseudo':
			return pseudo_random_choice
		elif method_name == 'true':
			return random_choice
		else:
			raise ValueError("Transition method name unknown. Please choose 'pseudo' or 'true' instead.")

if __name__ == "__main__":
	#----------------------------------- turn into args
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataroot', type=str, default='data/dylan')
	parser.add_argument('--output_dir', type=str, default="./graph")
	parser.add_argument('--window_size', type=int, default=10)
	parser.add_argument('--load_if_precomputed', action='store_true', default=False)
	parser.add_argument('--smpl_batch_size', type=int, default=64)
	parser.add_argument('--smpl_path', type=str, default='./human_dynamics/models/neutral_smpl_with_cocoplustoesankles_reg.pkl')
	parser.add_argument('--dist_threshold', type=float, default=0.5)
	args = parser.parse_args()
	#------------------------------------

	db = SubjectDatabase(args.dataroot, args.output_dir, args.window_size, args.smpl_path)

	if args.load_if_precomputed and os.path.exists(args.output_dir + '/edges.pkl'):
		print('----------')
		print('Loading precomputed graph')
		print('----------')
		db.load_graph()
	else:
		db.compute_graph(threshold=args.dist_threshold)

	if args.load_if_precomputed and os.path.exists(args.output_dir + '/transitions.pkl'):
		print('----------')
		print('Loading precomputed transitions')
		print('----------')
		db.load_transitions()
	else:
		db.compute_transitions(args.smpl_batch_size)
	seq = db.creative_walk()
	animate_sequence(seq, 'creative_walk', save_images=False)


