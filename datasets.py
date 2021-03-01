# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import sklearn.preprocessing
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D

"""
Dataset generator object
"""

class DatasetType(Enum):
    CROSS = 0
    HORSESHOE = 1
    CLOUDS = 2
    VEHICLE = 3 # https://archive.ics.uci.edu/ml/datasets/Statlog+%28Vehicle+Silhouettes%29
    LETTER = 4 # https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    AUSTRA = 5 # https://archive.ics.uci.edu/ml/datasets/Statlog+%28Australian+Credit+Approval%29
    WDBC = 6 # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

DATAFOLDER = os.path.join('..','datasets')


class Datagen:
    """
    Dataset generator object
    """
    
    def __init__(self, dataset_type, classlist0, classlist1, datafolder=DATAFOLDER):
        """
        Inputs:
            dataset_type: dataset type from DatasetType
            classlist0: list of labels for superclass 0
            classlist1: list of labels for superclass 1
            datafolder: set root folder where datasets live
        """
        
        self.dataset_type = dataset_type
        self.datafolder = datafolder

        self.classlist0 = classlist0
        self.classlist1 = classlist1

        if self.dataset_type == DatasetType.CROSS:
            self._genData = self._genCross
            self.d = 2
        elif self.dataset_type == DatasetType.HORSESHOE:
            self._genData = self._genHorseshoe
            self.d = 2
        elif self.dataset_type == DatasetType.CLOUDS:
            self._genData =  self._genClouds
            self.d = 2
        elif self.dataset_type == DatasetType.VEHICLE:
            self._genData = lambda N: self._loadVehicle(self.classlist0, self.classlist1)
            self.d = 18
        elif self.dataset_type == DatasetType.LETTER:
            self._genData = lambda N: self._loadLetter(self.classlist0, self.classlist1)
            self.d = 16
        elif self.dataset_type == DatasetType.AUSTRA:
            self._genData = lambda N: self._loadAustra()
            self.d = 14
        elif self.dataset_type == DatasetType.WDBC:
            self._genData = lambda N: self._loadWDBC()
            self.d = 30

        self.scaler = None

    def genData(self, N=600, preprocess=False, train_idx=None, test_idx=None):
        """
        Universal dataset generator function
        
        N: number of points (total)
        preprocess: flag for zero-mean, unit coordinatewise variance preprocessing
        train_idx: index of training set
        test_idx: index of test set
        """

        X,Y = self._genData(N)

        # if not provided, generate random partition
        if train_idx is None or test_idx is None:
            Ndata = X.shape[0]
            idx_shuff = list(range(Ndata))
            np.random.shuffle(idx_shuff)
    
            train_idx = idx_shuff[0:Ndata//2]
            test_idx = idx_shuff[Ndata//2:]
            
        Xtrain = X[train_idx]
        Ytrain = Y[train_idx]
        Xtest = X[test_idx]
        Ytest = Y[test_idx]

        self.scaler = sklearn.preprocessing.StandardScaler().fit(Xtrain)

        if preprocess:
            Xtrain = self.scaler.transform(Xtrain)
            Xtest = self.scaler.transform(Xtest)

        return Xtrain,Ytrain,Xtest,Ytest,train_idx,test_idx

    def _genCross(self, N):
        """
        Generates 'cross' dataset
        
        inputs:
            N: number of datapoints
        outputs:
            Xgen: N x 2 dataset
            Ygen: N-length yabels
        """

        pcorner = 0.25 # total percentage of samples in corner distributions
        ncorner = int(pcorner*N/4) # number of samples per corner
        nmain0 = int((N - ncorner*4)/2)
        nmain1 = N - nmain0 - ncorner*4

        cross_side = 1 # cross side length
        center_offset = 0.5 # center offset length
        main_var = 0.01 # variance of each cluster
        corner_frac = 1/4 # fraction of variance in corners
        Xgen = np.vstack((np.random.multivariate_normal(cross_side*np.array([1,1]), main_var*corner_frac*np.eye(2), ncorner),
                      np.random.multivariate_normal(cross_side*np.array([1,-1]), main_var*corner_frac*np.eye(2), ncorner),
                      np.random.multivariate_normal(center_offset*np.array([0,-1]), main_var*np.eye(2), nmain0),
                      np.random.multivariate_normal(cross_side*np.array([-1,1]), main_var*corner_frac*np.eye(2), ncorner),
                      np.random.multivariate_normal(cross_side*np.array([-1,-1]), main_var*corner_frac*np.eye(2), ncorner),
                      np.random.multivariate_normal(center_offset*np.array([0,1]), main_var*np.eye(2), nmain1)
                      ))
        Ygen = np.concatenate((np.zeros(ncorner*2 + nmain0),np.ones(ncorner*2 + nmain1)))

        assert(len(Xgen) == N and len(Ygen) == N)

        return Xgen,Ygen

    def _genHorseshoe(self, N, origin=None):
        """"
        Generates 'horseshoe' dataset
        
        inputs:
            N: number of datapoints
        outputs:
            Xgen: N x 2 dataset
            Ygen: N-length yabels
        """
        V = np.array([[1,1],[1,-1]])/np.sqrt(2)

        varsmall = 0.1
        varlarge = 1
        L = np.diag([varsmall,varlarge])

        if origin is None:
            origin = 3*np.sqrt(varsmall)*np.array([-1,1])/np.sqrt(2)

        cov0 = V.dot(L).dot(V)
        cov1 = V.dot(np.rot90(L,2)).dot(V)
        Xgen00 = np.random.multivariate_normal(origin + np.array([1,1]),cov0,N//3)
        Xgen01 = np.random.multivariate_normal(origin + np.array([-1,-1]),cov0,N//3)
        Xgen1 = np.random.multivariate_normal(origin + np.array([1,-1]),cov1,(N-(N//3)*2))
        Xgen = np.vstack((Xgen00,Xgen01,Xgen1))

        Ygen = np.array([0]*((N//3)*2) + [1]*(N-(N//3)*2))

        return Xgen,Ygen

    def _genClouds(self, N):
        """
        Generates 'clouds' dataset
        
        inputs:
            N: number of datapoints
        outputs:
            Xgen: N x 2 dataset
            Ygen: N-length yabels
        """

        mu0 = np.ones(2) # mean for class 0
        mu1 = -np.ones(2) # mean for class 1
        cov0 = np.eye(2) # covariance class 0
        cov1 = np.eye(2) # covariance class 1

        Xgen = np.vstack((np.random.multivariate_normal(mu0,cov0,N//2),
                          np.random.multivariate_normal(mu1,cov1,N-N//2)))
        Ygen = np.array([0]*(N//2) + [1]*(N-N//2))

        return Xgen,Ygen

    def _loadVehicle(self, classlist0, classlist1):
        """
        Loads vehicle recognition dataset. Possible classes: saab, opel, bus, van
        """
        
        datapath = os.path.join(self.datafolder,'UCI','vehicle');
        files = ['xaa.dat','xab.dat','xac.dat','xad.dat',
                         'xae.dat','xaf.dat','xag.dat','xah.dat','xai.dat']

        X = None
        Y = None
        for f in files:
            fullpath = os.path.join(datapath,f)
            df = pd.read_csv(fullpath,sep=' ',header=None,usecols=list(range(19)))
            arr = df.to_numpy()

            thisLabels = arr[:,-1].tolist()

            thisY = [0 if lab in classlist0 else 1 if lab
                in classlist1 else None for lab in thisLabels]

            thisX = np.array([arr[ii,:-1].astype('double') for ii in range(len(thisY))
                if thisY[ii] is not None])

            thisY = np.array(list(filter(lambda lab: lab is not None, thisY)))

            if X is None:
                X = thisX
            else:
                X = np.vstack((X,thisX))

            if Y is None:
                Y = thisY
            else:
                Y = np.concatenate((Y,thisY))

        return X,Y
    
    def _loadAustra(self):
        """
        Loads australian credit dataset.
        """
        
        datapath = os.path.join(self.datafolder,'UCI','austra');
        files = ['australian.dat']

        X = None
        Y = None
        for f in files:
            fullpath = os.path.join(datapath,f)
            df = pd.read_csv(fullpath,sep=' ',header=None,usecols=list(range(15)))
            arr = df.to_numpy()

            thisY = arr[:,-1].astype('int')
            thisX = arr[:,:-1].astype('double')

            if X is None:
                X = thisX
            else:
                X = np.vstack((X,thisX))

            if Y is None:
                Y = thisY
            else:
                Y = np.concatenate((Y,thisY))

        return X,Y

    def _loadWDBC(self):
        """
        Loads Wisconsin Breast Cancer dataset
        """
        
        datapath = os.path.join(self.datafolder,'UCI','wdbc');
        files = ['wdbc.data']

        X = None
        Y = None
        for f in files:
            fullpath = os.path.join(datapath,f)
            df = pd.read_csv(fullpath,sep=',',header=None,usecols=list(range(32)))
            arr = df.to_numpy()

            thisLabels = arr[:,1].tolist()

            thisY = np.array([0 if lab == 'M' else 1 for lab in thisLabels])
            thisX = arr[:,2:].astype('double')
            
            if X is None:
                X = thisX
            else:
                X = np.vstack((X,thisX))

            if Y is None:
                Y = thisY
            else:
                Y = np.concatenate((Y,thisY))

        return X,Y
    
    def _loadLetter(self, classlist0, classlist1):
        """
        Loads letter recognition dataset. Possible classes: letters A-Z
        """
        
        datapath = os.path.join(self.datafolder,'UCI','letter');
        files = ['letter-recognition.data']

        X = None
        Y = None
        for f in files:
            fullpath = os.path.join(datapath,f)
            df = pd.read_csv(fullpath,sep=',',header=None,usecols=list(range(17)))
            arr = df.to_numpy()

            thisLabels = arr[:,0].tolist()

            thisY = [0 if lab in classlist0 else 1 if lab
                in classlist1 else None for lab in thisLabels]

            thisX = np.array([arr[ii,1:].astype('double') for ii in range(len(thisY))
                if thisY[ii] is not None])

            thisY = np.array(list(filter(lambda lab: lab is not None, thisY)))

            if X is None:
                X = thisX
            else:
                X = np.vstack((X,thisX))

            if Y is None:
                Y = thisY
            else:
                Y = np.concatenate((Y,thisY))

        return X,Y

    def debugPlot(self, X, Yvec, ax, linewidth=1.0, msize=30,
                  c0='r', c1='b', alpha_base=1.0, alpha_top=1.0,
                  outidx=None, outcolors=None, fill=False):
        """
        Plotting for debugging
        """
        
        idx0 = [ii for ii in range(len(Yvec)) if Yvec[ii]==0]
        idx1 = [ii for ii in range(len(Yvec)) if Yvec[ii]==1]

        Xmin = np.min(X,axis=0)
        Xmax = np.max(X,axis=0)
        x0range = np.array([Xmin[0],Xmax[0]])
        x1range = np.array([Xmin[1],Xmax[1]])
            
        plt.sca(ax)
        
        if outidx is not None and outcolors is not None:
            assert len(outcolors)==len(outidx)
            zip0 = [tup for tup in zip(outidx,outcolors) if tup[0] in idx0]
            zip1 = [tup for tup in zip(outidx,outcolors) if tup[0] in idx1]
            
            outidx0, outcolors0 = zip(*zip0)
            outidx1, outcolors1 = zip(*zip1)
                
        if X.shape[1]==2:
            
            plt.scatter(X[idx0,0], X[idx0,1], s=msize, facecolors=c0,
                        edgecolors='none', alpha=alpha_base)
            plt.scatter(X[idx1,0], X[idx1,1], s=msize, facecolors=c1,
                        edgecolors='none', alpha=alpha_base)
            
            if outidx is not None and outcolors is not None:
                
                if fill:
                    plt.scatter(X[outidx0,0], X[outidx0,1], s=msize, c=outcolors0,
                                edgecolors='none', linewidths=linewidth, alpha=alpha_top)
                    plt.scatter(X[outidx1,0], X[outidx1,1], s=msize, c=outcolors1,
                                edgecolors='none', linewidths=linewidth, alpha=alpha_top)
                else:
                    plt.scatter(X[outidx0,0], X[outidx0,1], s=msize, facecolors='none',
                                edgecolors=outcolors0, linewidths=linewidth, alpha=alpha_top)
                    plt.scatter(X[outidx1,0], X[outidx1,1], s=msize, facecolors='none',
                                edgecolors=outcolors1, linewidths=linewidth, alpha=alpha_top)
            
            ax.set_xlim(x0range)
            ax.set_ylim(x1range)
            ax.set_aspect('equal', adjustable='box')
            
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            
            
if __name__ == '__main__':
    
    # working example
    datatype = DatasetType.VEHICLE
    classlist0=['saab','opel']
    classlist1=['bus','van']
    datagen = Datagen(datatype, classlist0=classlist0, classlist1=classlist1)
    Xtrain,Ytrain,Xtest,Ytest,_,_ = datagen.genData(0, preprocess=True)