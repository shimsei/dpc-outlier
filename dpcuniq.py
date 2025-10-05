# Compute local outlier probability for doctors
import numpy as np
from scipy.spatial.distance import cdist
#from math import *
from scipy.special import erf
from numpy import random
#import matplotlib.pyplot as plt
#from tempfile import mkdtemp
#import os.path as path
from multiprocessing import cpu_count
import pathos as pa

def cluster(x, sigma = 2, percent = 3, logrho = False, distfunc = 'euclidean', kernel = 'cutoff', by_delta = False, fig = False, dc = 0, ncut = 0, rhomin = 0, dmin = 0, nclust = 0, add_goop=True, normalize = False):
	"""
	runs density peak clustering (Rodriguez and Laio)
	Parameters
	x: ndarray
	   Data matrix x (samples x features)
	sigma: float. factor of standard deviation to consider outliers
	percent: float. percentage of data to consider for computing the local density
	logrho: bool. if True computes the logarithm of densities
			if False computes the densities
	distfunc: string. distance function of cdist
	kernel: string. 'cutoff' for threshold to compute densities. If ncut is positive integer it will use ncut neighbours. 
			 If ncut is zero it will compute it by percent variable.


	by_delta: bool.

	"""
	xx, xx_ind, xx_rind = np.unique(x, return_index = True, return_inverse=True, axis=0)
	print(x)
	nd = len(x[:,0]) # n data
	nf = len(x[0,:]) # n features
	ndxx = len(xx[:,0]) # n unique data
	print("input has ", ndxx, " unique elements")
	print("n data: ", nd, "n features: ", nf)

	d_nonzero_min = np.zeros(nf)
	rho = np.zeros(nd) # density
	delta = np.zeros(nd) # distance to nearest point with higher density
	nneigh = np.zeros(nd).astype(int)
	rhoxx = np.zeros(ndxx) # density of unique points
	deltaxx = np.zeros(ndxx) # distance to nearest point with higher density (for unique points)
	nneighxx = np.zeros(ndxx).astype(int)
	cl = -np.ones(nd).astype(int) # assigned cluster
	plof = np.zeros(nd) # probability of local outlier factor
	pdist = np.zeros(nd) 
	loop = np.zeros(nd)

	if ndxx == 1:
		cl = cl*0
		return loop, cl, rho, delta

	ncpu = cpu_count() - 2

	if dc == 0 and kernel == 'cutoff' and nclust != 1:
		if ncut == 0:
			ncut = nd*percent/100
		corr_avn = False # true if the avg number of neighbors is almost equal to ncut
		coeff1 = 1.2
		coeff2 = 0.05
		r = random.randint(ndxx, size=10) #
		dc = np.mean(cdist(xx,xx[r,:].reshape(-1,nf), distfunc))
		print(dc)
		pre_dc = dc -1
		pre_pre_dc = dc -2
		d2 = np.zeros(nd)
		times = 500

		if dc == 0:
			cl = cl*0
			return loop, cl, rho, delta
		print(xx[1,:].reshape(-1,nf))
		def find_min_max(i):
			d = cdist(xx,xx[i,:].reshape(-1,nf), distfunc)
			maxd2 = np.max(d)
			mind2 = np.min(d[d != 0])
			return [mind2, maxd2]
		inputs = range(ndxx)
		p = pa.pools.ProcessPool(ncpu)
		res = p.map(find_min_max, inputs)
		minmax = np.zeros((ndxx,2))
		minmax[:,0] = [i[0] for i in res]
		minmax[:,1] = [i[1] for i in res]
		p.close()
		p.join()
		p.clear()
		maxd = np.max(minmax[:,1])
		mind = np.min(minmax[:,0])
		print("min and max: ", mind, maxd)
		mind = np.abs(mind)
		r = random.randint(nd, size=times)
		r = xx_rind[r]

		def len_rand_neighbours(i):
			d1 = cdist(xx, xx[r[i],:].reshape(-1,nf), distfunc)
			d2 = d1[xx_rind]
			nnt2 = len(np.where(d2 <= dc)[0])
			return nnt2

		inputs = range(times)

		print("estimate dc")
		while corr_avn == False:
			p = pa.pools.ProcessPool(ncpu)
			res = p.map(len_rand_neighbours, inputs)
			nnt = np.mean(res)
			p.close()
			p.join()
			p.clear()
			print('..........',dc,nnt,coeff2,coeff1,ncut)
			if nnt > ncut*coeff1:
				pre_pre_dc = pre_dc
				pre_dc = dc
				dc = dc/(1+coeff2)
				if dc <= mind:
					corr_avn = True
				elif dc == pre_pre_dc:
					coeff2 = coeff2/2
					dc = dc/(1+coeff2)
				elif coeff2 < 0.001:
					corr_avn = True

			elif nnt < ncut/coeff1:
				pre_pre_dc = pre_dc
				pre_dc = dc
				dc = dc*(1+coeff2)
				if coeff2 < 0.001:
					corr_avn = True
			else:
				corr_avn = True

		print('dc = ', dc, 'avg_n = ', nnt)
	elif nclust == 1:
		r = random.randint(ndxx, size=10)
		dc = np.mean(cdist(xx,xx[r,:].reshape(-1,nf), distfunc))
		print('Select all data as one cluster, ', 'dc = ', dc)

	def find_rho_gaussian(i):
		d1 = cdist(xx,xx[i,:].reshape(-1,nf), distfunc)
		d2 = d1[xx_rind]
		rhoo = np.sum(np.exp(-np.square(d2)/(dc*dc)))
		return rhoo

	def find_rho_cutoff(i):
		d1 = cdist(xx,xx[i,:].reshape(-1,nf), distfunc)
		d2 = d1[xx_rind]
		rhoo = len(np.where(d2 <= dc)[0]) - 1
		return rhoo

	rho = np.zeros(nd)
	inputs = range(ndxx)

	p = pa.pools.ProcessPool(ncpu)
	if kernel == 'gaussian':
		if dc == 0:
			dc = 0.07
		print('Computing density with gaussian kernel of radius: ', dc)
		res = p.map(find_rho_gaussian, inputs)
	elif kernel == 'cutoff':
		print('Computing density with cutoff kernal of radius: ', dc)
		res = p.map(find_rho_cutoff, inputs)
	rhoxx = np.array(res)
	rho = rhoxx[xx_rind]
	p.close()
	p.join()
	p.clear()

	ordrho = np.zeros(nd)
	ordrho = np.argsort(-rho)

	ordrhoxx = np.zeros(ndxx)
	ordrhoxx = np.argsort(-rhoxx)
	deltaxx[ordrhoxx[0]] = -1
	nneighxx[ordrhoxx[0]] = -1

	delta[ordrho[0]] = -1
	nneigh[ordrho[0]] = -1

	print('calculate delta')

	def find_delta(i):
		ind = np.arange(i).astype(int)
		d1 = cdist(xx[ordrhoxx[ind],:].reshape(-1,nf),xx[ordrhoxx[i],:].reshape(-1,nf), distfunc)
		deltaa = np.amin(d1)
		j = ordrhoxx[np.argmin(d1)]
		k = np.where(xx_rind == j)[0]
		nneighh = k[0]
		return [deltaa, nneighh]
	inputs = range(1,ndxx)
	p = pa.pools.ProcessPool(ncpu)
	res = p.map(find_delta, inputs)
	deltaxx[ordrhoxx[1:]] = np.array([i[0] for i in res])
	nneighxx[ordrhoxx[1:]] = np.array([i[1] for i in res])
	p.close()
	p.join()
	p.clear()

	deltaxx[ordrhoxx[0]] = np.max(deltaxx)*1.000000001
	nneighxx[ordrhoxx[0]] = ordrhoxx[0]
	nneigh = xx_ind[xx_rind]
	nneigh[xx_ind] = nneighxx
	delta[xx_ind] = deltaxx

	if normalize == True:
		gamma = rho/np.max(rho)*delta/np.max(delta)
	else:
		gamma = np.multiply(delta,rho)

	dcut = dmin #np.median(delta)
	print("distance cut = ", dcut)
	#gamma = rho
	if logrho == True:
		gamma = np.multiply(delta,np.log(rho+1))

	if nclust == 0:
		if by_delta == True:
			dcut = dmin #np.max(np.std(delta) + np.mean(delta),dmin)
			nclust = len(np.where(delta >= dcut))
		else:
			nclust = len(np.where((gamma > (sigma * np.std(gamma) + np.mean(gamma))) & (delta >= dcut) & (rho >= rhomin))[0])
		if nclust == 0:
			nclust = 1

	print('find', nclust, 'clusters')
	# find cluster centers
	icl = np.zeros(nclust).astype(int)
	dhcenter = np.zeros(nclust)

	if by_delta == True:
		ordgamma = np.argsort(-rho)
	else:
		ordgamma = np.argsort(-gamma)
	j = 0

	print("available cluster center points with min delta, and min rho ", len(delta[(delta >= dcut) & (rho >= rhomin)]))
	for count in range(nclust):
		peak_found = False
		while peak_found == False:
			maxloc = ordgamma[j]
			if delta[maxloc] >= dcut and rho[maxloc] >= rhomin:
				peak_found = True
				print("found peak for clust ", count)
			else:
				j += 1
		icl[count] = maxloc
		j += 1
		dhcenter[count] = delta[maxloc]

	for ii in range(nclust):
		cl[icl[ii]] = ii
		print("clust ", ii, "center ", icl[ii], "clust ", cl[icl[ii]])

	# assignation
	for i in range(ndxx):
		j = ordrhoxx[i]
		ind = np.where(xx_rind == j)
		current_clust = np.max(cl[ind])
		if current_clust == -1:
			current_clust = np.max(cl[nneigh[ind]])
		cl[ind] = current_clust

	# calculate average density for each cluster
	density = np.zeros(nclust)
	nocc = np.zeros(nclust)
	peak_density = np.zeros(nclust)
	ind = np.where(cl == -1)[0]
	print("size unassigned: ", len(ind))
	for ii in range(nclust):
		ind = np.where(cl == ii)[0]
		nocc[ii] = len(ind)
		density[ii] = np.mean(rho[ind])
		peak_density[ii] = rho[icl[ii]]
		#print('clust ', ii, 'size ', nocc[ii],'density ', density[ii])

	# reorder by size
	cind = np.argsort(-nocc)
	jj = 0
	sorted_cl = np.zeros(nd)

	for ii in cind:
		ind = np.where(cl == ii)[0]
		sorted_cl[ind] = jj
		jj = jj +1

	for ii in range(nclust):
		ind = np.where(sorted_cl == ii)[0]
		jj = cl[ind[0]]
		print(ii, ': clust', jj, 'size', nocc[jj], 'mean density ', density[jj], 'peak density ', peak_density[jj], 'peak delta ', dhcenter[jj])

    # calculate PLOF
	print('calculate LoOP')
	goop = np.zeros(nclust)
	pgof = np.zeros(nclust)
	mean_pdist = np.zeros(nclust)
	for ii in range(nclust):
		pgof[ii] = (1/nocc[ii])/(1/(nd/nclust)) - 1
	npgof = sigma * np.sqrt(np.mean(np.square(pgof)))

	for ii in range(nclust):
		if npgof == 0:
			goop[ii] = 0
		else:
			goop[ii] = np.maximum(0,erf(pgof[ii]/(npgof*np.sqrt(2))))

	clxx = cl[xx_ind]
	pdistxx = np.zeros(ndxx)
	for i in range(ndxx):
		d1 = cdist(xx,xx[i,:].reshape(-1,nf), distfunc)
		d2 = d1[xx_rind]
		ii = clxx[i]
		ind = np.where(cl == ii)[0]
		pdistxx[i] = sigma * np.sqrt(np.sum(np.square(d2[ind]))/nocc[ii])

	pdist = pdistxx[xx_rind]
	for ii in range(nclust):
		ind = np.where(cl == ii)[0]
		mean_pdist[ii] = np.mean(pdist[ind])

	# to be tested
	#ind = np.where(cl > -2)[0]
	plof[:] = pdist[:]/mean_pdist[cl[:]] - 1
	#for i in range(nd):
		##plof[i] = pdist[i]/np.mean(pdist) - 1
		#plof[i] = pdist[i]/mean_pdist[cl[i]] - 1
	print(pdist)
	print(mean_pdist[cl[:]])
	print(plof)
	nplof = sigma * np.sqrt(np.mean(np.square(plof)))
	if add_goop == False:
		goop = goop * 0

	# to be tested
	for ii in range(nclust):
		ind = np.where(cl == ii)[0]
		if mean_pdist[ii] == 0:
			loop[ind] = goop[ii]
		else:
			loop[ind] = np.maximum(0, erf(plof[ind]/(nplof * np.sqrt(2))))
			loop[ind] = loop[ind] * (1 - goop[ii]) + goop[ii]
	#for i in range(nd):
	#	if mean_pdist[cl[i]] == 0:
	#		loop[i] = 0 + goop[cl[i]]
	#	else:
	#		loop[i] = np.maximum(0,erf(plof[i]/(nplof*sqrt(2)))) * (1 - goop[cl[i]]) + goop[cl[i]]

	print(goop)
	ind = np.where(loop > 0.9)[0]
	print("n outlierness > 0.9 ", len(ind))

	if fig == True:
		plt.subplot(211)
		plt.scatter(rho, delta)
		for ii in range(nclust):
			plt.scatter(rho[icl[ii]], delta[icl[ii]])
		plt.subplot(212)
		plt.scatter(x[ind,0],x[ind,1], s = (loop[ind]*20)**2, alpha=0.2)
		for ii in range(nclust):
			ind = np.where(cl == ii)[0]
			plt.scatter(x[ind,0], x[ind,1], alpha = 0.5, s=16)
		#plt.savefig('../data/dpc.png', format='png')
		plt.show()
	return loop, sorted_cl, rho, delta
