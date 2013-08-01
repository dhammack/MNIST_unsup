#python codez
import numpy as np
from sklearn import preprocessing as pp
from sklearn.cluster import KMeans as kmeans
from numpy.linalg import norm as norm
import itertools as it
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation as cv
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import random



def loadtrain(path, max=10000):
	data = [] #matrix without labels
	classes = []
	with open(path, 'r') as f:
		next(f) #skips the header
		for j,line in enumerate(f): 
			items = [float(i) for i in line.replace('\n','').split(',')]
			data.append(items[1:]) #skip the label
			classes.append(int(items[0])) #label
			if j > max: break
			
	datamat = np.matrix(data)
	classes = np.array(classes)
	return datamat,classes

def get_patches(X,sub_size,frac=0.5,orig_size=28):
	Xn = []
	for img in X:
		img2d = np.reshape(img, (orig_size, orig_size))
		imgs = image.extract_patches_2d( img2d, patch_size=(sub_size,sub_size),
											max_patches=frac)
		Xn.extend(imgs)
	
	Xn = np.array(Xn)
	Xn = np.reshape(Xn, (-1, sub_size ** 2) )
	return Xn
	
def simpleclassifier(X,y):
	
	model = SGDClassifier(loss='hinge',penalty='l2',shuffle=True)
	scores = cv.cross_val_score(model, X, y, cv=2)
	#model.fit(X,y)
	print scores.mean()
	
#maps R^n -> R^k and generates a sparse representation
#in the k dimensional subspace (k << n)
#we use K-means (triangle) for the feature mapping
def cluster_map(X, k, itr):
	mapper = kmeans(n_clusters=k, n_jobs=-1, n_init=itr) #only run once (weak comp)
	Xnew = mapper.fit_transform(X)
	
	#now apply the triangle method for sparsity
	#basically, it says to set to zero all distances which are 
	#further than the distance to the total mean
	x_mean = X.mean(axis=0) 
	#if the distance from the centroid is > dist to mean, set to 0
	transformeddata = []
	for x_old, x_new in it.izip(X,Xnew):
		#we can compute dist to mean as
		disttomean = norm(x_old-x_mean)
		#now we can compare that with each centroid dist
		transformedrow = [max(disttomean-z,0.0) for z in x_new]
		transformeddata.append(transformedrow)
	
	return np.matrix(transformeddata), mapper.cluster_centers_
	
def whiten(X,components): #whiten the data, return(data, PCA object)
	X = X - X.mean(axis=0)
	whitener = PCA(whiten=True, n_components = components)
	Xnew = whitener.fit_transform(X)
	Xnew = np.reshape(Xnew, (X.shape[0],-1))
	return Xnew, whitener
	
#prints the learned features
def print_feats(filters,imgsize=7,vert=7,horiz=7):
	plt.figure('Features')
	for i, f in enumerate(filters):
		plt.subplot(vert, horiz, i + 1)
		#print f.shape
		plt.imshow(f.reshape(imgsize, imgsize), cmap="gray")
		plt.axis("off")
	plt.show()
	
	
if __name__ == '__main__':
	
	pca_comps = 200 #components to use for PCA/whitening
	img_subsize = 12 #nxn
	train_ex_ct = 4000 #limit on training examples to use (comp weakness)
	display_feats = (10,50) # num of features is x1 * x2
	full_img_size = 28 #nxn 
	frac_of_subimages = 0.4 #fraction of possible subimages to use for training
	feature_detectors = display_feats[0] * display_feats[1]
	kmeans_iterations = 7
	L2_features = (10,20) #16 features
	L2_ft_ct = L2_features[0] * L2_features[1]
	
	print 'loading data...'
	X,y = loadtrain('train.csv',max=train_ex_ct)
	
	print 'subsampling images...'
	Xsub = get_patches(X,sub_size = img_subsize,
						frac = frac_of_subimages,
						orig_size = full_img_size)
	
	print 'whitening images, using', pca_comps, 'PCA components...'
	Xwhite, whitener = whiten(Xsub, pca_comps)
	
	print 'running K-means, k =',feature_detectors,', to detect features...'
	X_L1, L1_centroids = cluster_map(Xwhite, feature_detectors, kmeans_iterations)
	
	print 'visualizing detected features...'
	orig_clusters = [whitener.inverse_transform(c) for c in L1_centroids]
	#print_feats(orig_clusters, imgsize=img_subsize,vert=display_feats[0],horiz=display_feats[1])
	
	#now we've learned some low-level features
	#what if we try to learn digits?
	
	print 'running K-means again, trying to learn digits...'
	X_L2, L2_centroids = cluster_map(X_L1, L2_ft_ct, kmeans_iterations)
	
	print 'projecting to pixel space...'
	#assume dig_centroids tells us the location in the l1 feature space
	L1_proj = []
	print 'L2_centroids:',L2_centroids.shape, 'L1_centroids:',L1_centroids.shape
	
	L1_proj = np.dot(L2_centroids, L1_centroids)
	

	print 'trying to visualize'
	orig_clusters_L1 = [whitener.inverse_transform(c) for c in L1_proj]
	#TEMP: remove the top one of these
	print_feats(orig_clusters, imgsize=img_subsize,vert=display_feats[0],horiz=display_feats[1])
	print_feats(orig_clusters_L1, imgsize=img_subsize,
				vert=L2_features[0],horiz=L2_features[1])
	