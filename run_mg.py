import heapq
import time
import pickle
import sys
import os
import numpy as np
import argparse
import skimage.io as skio
import matplotlib.pyplot as plt
from motion_graph import SubjectDatabase
from viz import draw_pose_openpose, animate_sequence
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict

def copy_path(path):
	p = Path(path.db)
	p.path = path.get_path().copy()
	p.crosses = path.crosses.copy()
	p.backs = path.backs.copy()
	p.skips = path.skips.copy()
	p.cost = path.get_cost()
	p.size = path.size
	p.seen = path.seen.copy()
	return p

class Path():
	def __init__(self, db):
		self.path = []
		self.cost = 0
		self.db = db
		self.size = 0
		self.crosses = [-999]
		self.backs = [-999]
		self.skips = [-999]
		self.seen = {}

	def __repr__(self):
		return"<Path: "+ str(self.cost) + ", " + str(self.path) + ">"

	def _additional_cost(self, new_node, last_node, last_interesting_edge, size):
		cost = 0
		edge_type = None
		if self.db.nodes[new_node].name != self.db.nodes[last_node].name:
			cost = 300/(size - last_interesting_edge)
			edge_type = "cross"
		elif self.db.nodes[new_node].index < self.db.nodes[last_node].index:
			cost = 300/(size - last_interesting_edge)
			edge_type = "back"
		elif self.db.nodes[new_node].index > self.db.nodes[last_node].index + self.db.window_size:
			cost = 600/(size - last_interesting_edge)
			edge_type = "skip"
		else:
			cost = min((size-last_interesting_edge)/60, 10)
		return cost, edge_type

	def additional_cost(self, node):
		"""Mathematical scheme for "creativity". Encourage interesting paths (defined as 
		those that include skip, cross, and back edge connections) while also penalizing
		'too much' exploration such that the end result is incomprehensible. This is done by
		scaling the novelty cost of the interesting path by how long ago the last interesting
		edge was added. By contrast, a normal forward edge has greater value when it follows
		soon after a recent interesting edge. 
		"""
		last_interesting_edge = max([self.crosses[-1], self.backs[-1], self.skips[-1]])
		return self._additional_cost(node, self.path[-1], last_interesting_edge, self.size)

	def add_node(self, node):
		if self.size > 0:
			cost, edge_type = self.additional_cost(node)
			self.cost += cost
			if edge_type == "cross":
				self.crosses.append(self.size)
			elif edge_type == 'back':
				self.backs.append(self.size)
			elif edge_type == 'skip':
				self.skips.append(self.size)
		self.size += 1
		self.path.append(node)
		self.seen[node] = True

	def get_cost(self):
		return self.cost

	def recompute_data(self):
		self.cost = 0
		self.crosses = [-999]
		self.backs = [-999]
		self.skips = [-999]
		self.seen = {}
		self.seen[self.path[0]] = True

		last_interesting_edge = max([self.crosses[-1], self.backs[-1], self.skips[-1]])
		for i in range(1, self.size):
			cost, edge_type = self._additional_cost(self.path[i], self.path[i-1], last_interesting_edge, i)
			self.cost += cost
			if edge_type == "cross":
				self.crosses.append(i)
			elif edge_type == 'back':
				self.backs.append(i)
			elif edge_type == 'skip':
				self.skips.append(i)
			last_interesting_edge = max([self.crosses[-1], self.backs[-1], self.skips[-1]])

			self.seen[self.path[i]] = True

	def get_curr_node(self):
		return self.path[-1]

	def get_path(self):
		return self.path

	def __lt__(self, other):
		return self.cost < other.get_cost()


#-----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='data/dylan')
parser.add_argument('--output_dir', type=str, default="./graph")
parser.add_argument('--window_size', type=int, default=10)
parser.add_argument('--input_method', type=str, default='random') #Options: random, screen, manual
parser.add_argument('--smpl_path', type=str, default='./human_dynamics/models/neutral_smpl_with_cocoplustoesankles_reg.pkl')
parser.add_argument('--n', type=int, default=90)
parser.add_argument('--m', type=int, default=30)
parser.add_argument('--stop_threshold', type=float, default=20000) #20000 / (30fps * 10 base cost) ~ 60 second video
parser.add_argument('--savedir', type=str, default="constrained_walk")
args = parser.parse_args()
#------------------------------------

'''
LOAD DATABASE
'''

db = SubjectDatabase(args.dataroot, args.output_dir, args.window_size, args.smpl_path)

db.load_graph()
if not os.path.exists(args.output_dir + '/edges.pkl'):
	print("Please compute the motion graph first.")
	sys.exit(0)

db.load_transitions()
if not os.path.exists(args.output_dir + '/transitions.pkl'):
	print("Please finish computing the motion graph first.")
	sys.exit(0)

'''
GET AND SAVE START/END POSES
'''

edges_crs = csr_matrix(db.edges)
n_components, labels = connected_components(csgraph=edges_crs, directed=True, return_labels=True, connection='strong')

if args.input_method == 'random':
	start = db.nodes[db.sample_edge()[0]]
	end = db.nodes[db.sample_edge()[1]]
	start_index = start.node_index
	end_index = end.node_index

	print("Start: %s frame %d, End: %s frame %d" % (start.name, start.index, end.name, end.index))

elif args.input_method == 'screen':
	more = True
	while more:
		start = db.nodes[db.sample_edge()[0]]
		skio.imshow(draw_pose_openpose(start.joints_2d().flatten(), 640, 640))
		plt.title(start.name + " frame " + str(start.index))
		skio.show()
		more = bool(int(input("Input 1 to refresh, 0 to confirm. ")))
	print("Selected %s frame %d" % (start.name, start.index))
	print()

	more = True
	while more:
		end = db.nodes[db.sample_edge()[1]]
		skio.imshow(draw_pose_openpose(end.joints_2d().flatten(), 640, 640))
		plt.title(end.name + " frame " + str(end.index))
		skio.show()
		more = bool(int(input("Input 1 to refresh, 0 to confirm. ")))
	print("Selected %s frame %d" % (end.name, end.index))
	print()
	start_index = start.node_index
	end_index = end.node_index

elif args.input_method == 'manual':
	start_index = int(input("Start Node Index: "))
	end_index = int(input("End Node Index: "))

if not os.path.exists(args.savedir):
	os.makedirs(args.savedir)

A = draw_pose_openpose(db.nodes[start_index].joints_2d().flatten(), 640, 640)
B = draw_pose_openpose(db.nodes[end_index].joints_2d().flatten(), 640, 640)
skio.imsave(os.path.join(args.savedir, 'A.png'), A)
skio.imsave(os.path.join(args.savedir, 'B.png'), B)

if labels[start_index] != labels[end_index]:
	print("Warning: Specified frames are not a part of the same SCC, no path will be found.")

'''
SEARCH FOR PATH BETWEEN POSES
'''

print("Finding path betwen %d and %d" % (start_index, end_index))

start_path = Path(db)
start_path.add_node(start_index)

best_path = None
best_total_cost = float("inf")

pq = []
heapq.heappush(pq, (start_path.get_cost(), start_path))

start = time.time()

path_offset = 0

best_cost_ending_at = defaultdict(lambda: float("inf"))

while best_path is None:
	best_cost = float("inf")
	best_n_path = None
	while len(pq) > 0 and (best_n_path is None or best_n_path.get_cost() > args.stop_threshold):
		curr_cost, curr_path = heapq.heappop(pq)
		curr_node = curr_path.get_curr_node()

		if curr_node == end_index and curr_cost < best_total_cost: # found a complete path
			best_total_cost = curr_cost
			best_path = curr_path
		elif (curr_path.size - path_offset) >= args.n and curr_cost < best_cost: # found a complete n path
			best_cost = curr_cost
			best_n_path = curr_path
			print("found new best n path")
		elif curr_cost < best_cost and curr_cost < best_cost_ending_at[curr_node]: # keep searching
			best_cost_ending_at[curr_node] = curr_cost
			next_node_options = np.argwhere(db.edges[curr_path.get_curr_node()] == 1).flatten()
			for node in next_node_options:
				if node not in curr_path.seen: #prevent continuous loops
					extended_path = copy_path(curr_path)
					extended_path.add_node(node)
					heapq.heappush(pq, (extended_path.get_cost(), extended_path))

	print(best_n_path.get_cost())
	print()
	if best_n_path is None and best_path is None:
		print("No path found.")
		sys.exit(0)
	elif best_path is None:
		# start new search
		new_start = Path(best_n_path.db)
		new_start.path = best_n_path.get_path()[:args.m]
		new_start.size = len(new_start.get_path())
		new_start.recompute_data()

		pq = []
		heapq.heappush(pq, (new_start.get_cost(), new_start))
		path_offset += args.m
		best_cost_ending_at = defaultdict(lambda: float("inf"))


time_taken = time.time() - start
print("Total search time %f" % time_taken)

'''
SAVE AND ANIMATE
'''

if best_path is not None:
	f = open(os.path.join(args.savedir, 'path.txt'), 'w')
	for node_index in best_path.get_path():
		f.write(str(db.nodes[node_index]) + '\n')
	f.close()

	sequence = []
	node_sequence = best_path.get_path()
	sequence.append(db.nodes[node_sequence[0]].joints_2d())
	for i in range(1, len(node_sequence)):
		if db.transitions[node_sequence[i-1]][node_sequence[i]] is not None:
			sequence.append(db.transitions[node_sequence[i-1]][node_sequence[i]].joints_2d())
		sequence.append(db.nodes[node_sequence[i]].joints_2d())
	sequence = np.concatenate(sequence, axis=0)
	animate_sequence(sequence, args.savedir, save_images=True)
else:
	print("No path found.")