import os
import pickle
import numpy as np
import seaborn as sns

def save(output_path, filename, ob):
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	full_path = os.path.join(output_path, filename)
	f = open(full_path, 'wb')
	pickle.dump(ob, f)

def load(full_path):
	assert os.path.exists(full_path), "no file matching the path found"
	f = open(full_path, 'rb')
	return pickle.load(f)

def slerp(p1, p2, t):
	""" Spherical Linear Interpolation of Quaternions
	"""
	assert(p1.shape == p2.shape)
	p1 = p1.reshape(-1, 4)
	p2 = p2.reshape(-1, 4)

	p1 = p1/np.sqrt(np.sum(p1**2, axis=1)).reshape(-1, 1)
	p2 = p2/np.sqrt(np.sum(p2**2, axis=1)).reshape(-1, 1)

	dot_prods = np.sum(p1*p2, axis=1) #prevents puzzling read-only error

	negative = (dot_prods < 0)
	if np.sum(negative) > 0:
		p2[negative] *= -1
		dot_prods[negative] *= -1

	dot_threshold = 0.9995
	close_to_thre = (dot_prods > dot_threshold)

	res = np.empty(p1.shape)

	# just linearly interpolate if close
	if np.sum(close_to_thre) > 0:
		res[close_to_thre] = p1[close_to_thre] + t*(p2[close_to_thre]-p1[close_to_thre])
		n = np.sum(close_to_thre)
		res[close_to_thre] /= np.sqrt(np.sum(res[close_to_thre]**2, axis=1)).reshape(n, 1)

	# normal slerp
	theta_0 = np.arccos(dot_prods[~close_to_thre]).reshape(-1, 1)
	theta = t*theta_0
	sin_theta = np.sin(theta)
	sin_theta_0 = np.sin(theta_0)


	s2 = sin_theta / sin_theta_0
	s1 = np.cos(theta) - dot_prods[~close_to_thre].reshape(-1,1) * s2
	res[~close_to_thre] =  (s1 * p1[~close_to_thre]) + (s2 * p2[~close_to_thre])
	return res

def lerp(p1, p2, t):
	""" Normal Linear Interpolation
	"""
	assert(p1.shape == p2.shape)
	return t*p1 + (1-t)*p2

def alpha(p, k):
	"""Transition scheme used in Kovar et. al 2002
	"""
	if p <= -1:
		return 1
	if p >= k:
		return 0
	w = 2*((p+1)/k)**3 - 3*((p+1)/k)**2 + 1
	return w

# numpy re-implementation of the method from HMMR
def orth_proj_idrot(X, camera):
	camera = camera.reshape(-1, 1, 3)
	X_trans = X[:, :, :2] + camera[:, :, 1:]
	shape = X_trans.shape
	return (camera[:, :, 0] * X_trans.reshape(shape[0], -1)).reshape(shape)

def aa2quart(aa):
	if aa is None:
		return []
	aa = aa.reshape(-1, 3)
	quart = []
	for v in aa:
		if v.sum() == 0:
			quart.append(np.array([0,0,0,1]))
		else:
			magnitude = np.linalg.norm(v)
			if magnitude < 1e-6: #near origin, compute sinc function instead
				e_v = v * ((1/2) + (magnitude**2 / 48) ) 
			else: #compute unit vector as normal
				e_v = np.sin(magnitude/2) * v/magnitude
			quart.append(np.append(e_v, np.cos(magnitude/2)))
	return np.array(quart).flatten()

def quart2aa(quart):
	"""
	Args
	- quart: (24, 4) or (96, ) array of quarternions
	"""
	if quart is None:
		return []
	quart = quart.reshape(-1, 4)
	q_e = quart[:, :3]
	e = q_e/np.sqrt(np.sum(q_e**2, axis=1)).reshape(-1, 1)
	theta = 2* np.arccos(quart[:, -1]).reshape(-1, 1)
	return e * theta

# forces a non-trivial random choice
def pseudo_random_choice(choices, curr_node):
	n = len(choices)
	if n > 1:
		return np.random.choice(choices[choices != curr_node+1])
	else:
		return choices[0]

def random_choice(choices, curr_node):
	return np.random.choice(choices)