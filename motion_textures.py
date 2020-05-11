import argparse
import numpy as np
import os
import time
import pickle
from multiprocessing import Pool
from functools import partial 
from scipy.stats import multivariate_normal

def exponential_map(thetas):
	assert(thetas.shape == (72, ))
	thetas = thetas.reshape(24, 3)
	reparameterized = []
	for v in thetas:
		if v.sum() == 0:
			reparameterized.append(np.zeros(3))
		else:
			magnitude = np.linalg.norm(v)
			if magnitude < 1e-6: #near origin, compute sinc function instead
				e_v = v * ((1/2) + (magnitude**2 / 48) ) 
			else: #compute unit vector as normal
				e_v = np.sin(magnitude/2) * v/magnitude
			reparameterized.append(e_v)
	return np.array(reparameterized).flatten()

def rot2eul(R):
	"""
	Converts batch of rotation matrices R into euler angles
	Args:
		-R: (N x 3 x 3)

	Returns:
		-eul: (N X 3) 
	"""

	# we ignore second possible solutions because we prefer "smaller" angles

	R = R.reshape(-1, 9)
	theta1 = -np.arcsin(R[:, 6])
	# theta2 = np.pi - theta1
	psi1 = np.arctan2(R[:, 7]/np.cos(theta1), R[:, 8]/np.cos(theta1))
	# psi2 = np.arctan2(R[:, 7]/np.cos(theta2), R[:, 8]/np.cos(theta2))
	phi1 = np.arctan2(R[:, 3]/np.cos(theta1), R[:, 0]/np.cos(theta1))
	# phi2 = np.arctan2(R[:, 3]/np.cos(theta2), R[:, 0]/np.cos(theta2))

	gimbal_case_1 = R[R[:, 6] == -1]
	psi1[gimbal_case_1] = np.arctan2(R[:, 1], R[:, 2])
	phi1[gimbal_case_1] = 0

	gimbal_case_2 = R[R[:, 6] == 1]
	psi1[gimbal_case_2] = np.arctan2(-R[:, 1], -R[:, 2])
	phi1[gimbal_case_2] = 0

class Frame():
	def __init__(self, cams, skeleton, skeleton_posed, name="unknown", index=-1):
		assert(cams.shape[-1] == 3)
		assert(thetas.shape[-1] == 72)
		assert(betas.shape[-1] == 10)
		# self.thetas = thetas
		# self.betas = betas
		self.cams = cams
		# self.joints = joints 
		# self.kps = kps
		self.name = name
		self.index = index

		self.skeleton = skeleton
		self.skeleton_posed = skeleton_posed

		poses = rot_mat(skeleton, skeleton_posed)

		self.exp = exponential_map(rot2eul(poses))

	def __repr__(self):
		return "<" + self.name + " frame " + str(self.index) + ">"

class LDS():
	def __init__(self, C, W, A1, A2, D, B, latent_dim):
		self.C = C
		self.W = W #scipy.stats.mutlivariate normal
		self.A1 = A1
		self.A2 = A2

		self.v = multivariate_normal(mean=0, cov=1)
		self.D = D
		self.B = np.real(B)
		self.prior = None
		self.latent_dim = latent_dim

	def step(self, X): # X -> [x_t x_tm1]^T
		return None, None

	def add_to_prior(self, keyposes):
		if self.prior is None:
			self.prior = np.array([keyposes])
		else:
			self.prior = np.concatenate([self.prior, np.array([keyposes]) ])

	def probability(self, observations):
		d, T = observations.shape
		U, S, Vt = np.linalg.svd(observations)
		S = np.diag(S[:self.latent_dim])
		S = np.concatenate([S, np.zeros((self.latent_dim, T - self.latent_dim))], axis=1)
		X = (S @ Vt) 

		predicted_latent = (self.A1 @ X[:, 1:-1]) + (self.A2 @ X[:, :-2]) + self.D
		latent_noise = (X[:, 2:] - predicted_latent).T / self.B #(T-2)
		p = 1
		for latent_vec in latent_noise:
			p *= np.prod(self.v.pdf(latent_vec)) #(T-2, ) 
		obs_prob = self.W.pdf( ((self.C @ X) - observations).T ) #(T, )
		# keyposes = X[:, :2]

		# print(self.prior == keyposes)
		# p = (np.sum(self.prior == keyposes)//(2*D)) /(len(self.prior))
		# print(p)
		obs_prob[np.isnan(obs_prob)] = 1
		p *= np.prod(np.clip(obs_prob, 0, 1))
		# print(p)
		assert(0 <= p <= 1)
		return p

def load_data(root):
	frames = []
	f = os.path.join(root, 'hmmr_output.pkl')
	if os.path.exists(f):
		data = pickle.load(open(f, 'rb'))
		n_frames = len(data['kps'])
		for i in range(n_frames): 
			frame = Frame(data['cams'][i], data['poses'][i], name=f.split('/')[-2], index=i)
			frames.append(frame)
	return frames[:200]

def transition_matrix(labels, N_t):
	M = np.zeros((N_t, N_t))
	for i in range(N_t):
		for j in range(N_t):
			M[i, j] = np.sum((labels[:-1] == i) & (labels[1:] == j))
		total = np.sum(M[i])
		if total > 0:
			M[i] /= total
	return M

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def fit_LDS(segment, mask):
	"""
	segment - numpy array of shape (74, T)
	mask - numpy array of shape (T, )
	"""

	#convert translation to displacement from starting pose
	segment[-2:] -= segment[-2:, 0].reshape(2, 1)

	obs_dim, T = segment.shape
	U, S, Vt = np.linalg.svd(segment)

	latent_dim = 15
	C = U[:, :latent_dim]
	S = np.diag(S[:latent_dim])
	S = np.concatenate([S, np.zeros((latent_dim, T - latent_dim))], axis=1)
	X = mask*(S @ Vt)

	scaling = 1/(T-2)

	Q = [None, None, None]
	for i in range(3):
		Q[i] = np.sum( X[:, 2-i:T-i], axis=1).reshape(-1, 1)

	R = [[0, 0, 0], 
		 [0, 0, 0],
		 [0, 0, 0]]

	for i in range(3):
		for j in range(3):
			R[i][j]= np.einsum('ji,ki->jk', X[:, 2-i:T-i], X[:, 2-j:T-j]) - scaling * (Q[i] @ Q[j].T) # 1d @ 1d is automatically dot product

	first_term = np.block([R[0][0], R[0][1]])
	second_term = np.block([[R[1][1], R[2][1]], [R[1][2], R[2][2]]])
	A = first_term @ np.linalg.inv(second_term)
	D = scaling * (Q[0]  - (A[:, :latent_dim] @ Q[1]) - (A[:, latent_dim:] @ Q[2]))	
	BBT = scaling * (R[0][0]  - (A[:, :latent_dim] @ R[1][0]) - (A[:, latent_dim:] @ R[2][0]))

	# (BB^T)B = B(B^TB) = B * |B|^2 = lambda * B 
	# we can use the above relation to solve for B using eigenvalues
	vals, vecs = np.linalg.eig(BBT)
	lowest_error = float("inf")
	B = None
	val = None
	for i in range(latent_dim):
		if vals[i] != 0:
			eig_vec = vecs[:, i].reshape(-1, 1)
			error = np.sum(np.abs((eig_vec @ eig_vec.T) - BBT/vals[i])) 
			if error < lowest_error:
				B = np.real(eig_vec * np.sqrt(vals[i]+0j)) #sometimes best fit eigenvalue is negative, in which case this sets B equal to 0
				lowest_error = error
				val = vals[i]
	print(lowest_error)
	assert(B is not None), "Could not calculate corresponding B"
	W = multivariate_normal(np.zeros(obs_dim), np.cov(segment) + (np.eye(obs_dim) * 1e-8))
	return LDS(C, W, A[:, :latent_dim], A[:, latent_dim:], D, B.flatten(), latent_dim)

# E step
def inference(observations, T_min, textons, M):
	D, T = observations.shape
	N_t = len(textons)

	g = np.zeros((T, T)) # g[n, t] = max value of likelihood  when diving sequence ending at frame t into n segments
	e = np.zeros((T, T), dtype=np.int32) # e[n ,t] = label of last segment to achieve g[k ,t]
	f = np.zeros((T, T), dtype=np.int32) # f[n ,t] = start point of last segment to achieve g[k ,t]

	#initialization
	for t in range(T_min-1, T):
		for i in range(N_t):
			probability = textons[i].probability(observations[:, :t])
			if probability > g[0, t]:
				g[0, t] = probability
				e[0, t] = i

	# loop; for all possible segments and ending frames
	for n in range(1, T//T_min):
		for t in range(n*T_min - 1, T):
			# find the best possible additional segment
			for i in range(N_t):
				for b in range((n-1)*T_min, (t-T_min)):
					probability = textons[i].probability(observations[:, b:t])
					label = e[n-1, b-1]
					value = g[n-1, b-1] * probability * M[label, i]
					if value > g[n, t]:
						g[n, t] = value
						e[n, t] = i
						f[n, t] = b

	N_s = np.argmax(g[:(T//T_min), -1]) + 1
	final_prob = g[N_s - 1, -1]

	H = np.empty(N_s+1)
	L = np.empty(N_s)
	H[-1] = T
	L[-1] = e[N_s-1][-1]
	H[0] = 1
	for n in range(N_s-1, 0, -1):
		H[n] = f[n, (H[n+1]-1)]
		L[n-1] = e[-1, (H[n]-1)]
	return H, L, final_prob

# M step
def refit_to_labels(observations, labels, seg_points, N_t):
	N_s = len(labels)
	labels = labels.astype(np.int32)
	seg_points = seg_points.astype(np.int32)

	# gather all segments associated with each texton
	observations_per_texton = [None] * N_t
	masks = [None] * N_t
	for i in range(N_s):
		label = int(labels[i])
		if observations_per_texton[label] is not None:
			observations_per_texton[label] = np.concatenate([observations_per_texton[label], observations[:, seg_points[i]:seg_points[i+1]]], axis=1)
			# drop terms near the boundary of the segments
			mask[-2:] = 0 
			mask = np.append(mask, [np.zeros(2), np.ones(seg_points[i+1]-seg_points[i]+2)])
		else:
			observations_per_texton[label] = observations[:, seg_points[i]:seg_points[i+1]]
			mask = np.ones(seg_points[i+1]-seg_points[i])
	# refit textons
	textons = []
	for i in range(N_t):
		if masks[i] is not None:
			textons.append(fit_LDS(observations_per_texton[i], masks[i]))

	M = transition_matrix(labels, N_t)
	return textons, M

def learn_motion_texture(observations, T_min, init_threshold):
	print('----------')
	print('Greedily initializing')
	print('----------')

	# greedy initialization
	textons = []
	labels = []
	N_t = 0
	N_s = 0
	curr_index = 0
	T = len(observations)
	while curr_index < T:
		# create new LDS
		textons.append(fit_LDS(observations[:, curr_index:curr_index + T_min], np.ones(T_min)))
		N_t += 1
		fit_error = 1 -  textons[-1].probability(observations[:, curr_index:curr_index + T_min])
		print(fit_error)
		seg_len = T_min
		while fit_error < init_threshold and curr_index < T:
			seg_len += 1
			# textons[-1] = fit_LDS(observations[:, curr_index:curr_index + seg_len], np.ones(seg_len))
			fit_error += textons[-1].probability(observations[:, curr_index:curr_index + seg_len])
		labels.append(N_t - 1)
		N_s += 1
		curr_index += seg_len

		# try to associate existing LDS with remaining frames
		def find_best_fit(start_index):
			best_error = 1
			best_fit = None
			for t in range(N_t):
				fit_error = 1 - textons[t].probability(observations[:, start_index:start_index + T_min])
				if fit_error < best_error:
					best_error = fit_error
					best_fit = t
			return best_error, best_fit

		if curr_index >= T:
			break
		best_error, best_fit = find_best_fit(curr_index)
		while curr_index < T and best_error < init_threshold:
			# associate segment with existing texton
			curr_index += T_min
			labels.append(best_fit)
			N_s += 1

			# see if we can keep going
			best_error, best_fit = find_best_fit(curr_index)

	# initialize transition matrix
	M = transition_matrix(np.array(labels), N_t)

	print('----------')
	print('Starting EM loop')
	print('----------')

	# first E step
	start = time.time()
	seg_points, labels, energy = inference(observations, T_min, textons, M)
	time_taken = time.time() - start 
	print("Single inference step time: %f" % (time_taken))

	start = time.time()
	# loop until local convergence
	delta = float("inf")
	prev_energy = energy
	print()
	print(energy)
	while delta > 1e-3:
		# M step: fit new textons
		textons, M = refit_to_labels(observations, labels, seg_points, N_t)

		# E step
		seg_points, labels, energy = inference(observations, T_min, textons, M)
		print(energy)
		delta = np.abs(energy - prev_energy)
		prev_energy = energy

	time_taken = time.time() - start
	print("Total EM time: %f" % (time_taken))

	return labels, seg_points, textons, M


# L = numeric idx of the segment
# H = starting frame of the segment
# T = texton; LDS
# THETA = parameters of texton/LDS
parser = argparse.ArgumentParser()
parser.add_argument('--lds_error_threshold', type=float, default=0.5)
parser.add_argument('--data_root', type=str, default='data/dylan/eta')
parser.add_argument('--t_min', type=int, default=30)
args = parser.parse_args()

frames = load_data(args.data_root)
assert(len(frames) > 0), "No observations found, please double check data path"
print("T: %d" % len(frames))

observations = np.array([np.append(frame.exp, frame.cams[1:]) for frame in frames]) #(T, 74)

segment_labels, segmentation_points, textons, texton_distributions = learn_motion_texture(observations.T, args.t_min, args.lds_error_threshold)
f = open('motion_textures.pkl', 'wb')
pickle.dump((segment_labels, segmentation_points, textons, texton_distributions), f)