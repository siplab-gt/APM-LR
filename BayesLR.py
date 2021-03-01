#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.linear_model, sklearn.datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datasets import Datagen, DatasetType
import time

"""
Logistic regression learning object
"""

class BayesLR:

    """
    maintains posterior distribution and estimate for logistic regression
    """

    def __init__(self, X=None, Yvec=None, d=2, fit_intercept=False, Cw=1e2, Cb=1e2, Xpool=None, posttype='Variational', tol=1e-6):
        """
        Initialize class object
        Inputs:
            X: (Npoints,d) array of examples
            Yvec: (Npoints,) vector labels (0 or 1)
            d: data dimension
            fit_intercept: (Bool) if true, learns hyperplane offset
            Cw: coordinatewise variance on isotropic Gaussian weights prior. equivalent to
               inverse of l2 penalty
            Cb: coordinatewise variance on isotropic Gaussian bias prior. equivalent to
               inverse of l2 penalty
            Xpool: included for visualization purposes
            posttype: posterior approximation type {'Laplace','Variational'}
            tol: tolerance for variational approximation convergence
        """

        if X is None:
            self.initialize(X=None, Yvec=None, d=d, fit_intercept=fit_intercept, Cw=Cw, Cb=Cb, Xpool=Xpool, posttype=posttype, tol=tol)
        else:
            self.initialize(X=X, Yvec=Yvec, d=X.shape[1], fit_intercept=fit_intercept, Cw=Cw, Cb=Cb, Xpool=Xpool, posttype=posttype, tol=tol)

    def initialize(self, X=None, Yvec=None, d=1, fit_intercept=False, Cw=1e2, Cb=1e2, Xpool=None, posttype='Variational', tol=1e-6):
        """
        Reset class object
        Inputs:
            X: (Npoints,d) array of examples
            Yvec: (Npoints,) vector labels (0 or 1)
            d: data dimension
            fit_intercept: (Bool) if true, learns hyperplane offset
            Cw: coordinatewise variance on isotropic Gaussian weights prior. equivalent to
               inverse of l2 penalty
            Cb: coordinatewise variance on isotropic Gaussian bias prior. equivalent to
               inverse of l2 penalty
            Xpool: included for visualization purposes
            posttype: posterior approximation type {'Laplace','Variational'}
            tol: tolerance for variational approximation convergence
        """

        self.Cw = Cw
        self.Cb = Cb
        self.d = d
        self.X = X
        self.Xpool = Xpool
        self.Yvec = Yvec
        self.tol = tol
        self.posttype = posttype
        self.lr_weight = np.random.rand(d)/1000 # choose value close to (but not quite) 0 for stability

        # y = sign(lr_weight.T * x - lr_bias)
        if fit_intercept:
            self.lr_bias = np.random.rand()/1000
            self.invcov = np.diag(np.append((1.0/Cw)*np.ones(d),1.0/Cb))
        else:
            self.lr_bias = None
            self.invcov = np.diag((1.0/Cw)*np.ones(d))

        self.lr = sklearn.linear_model.LogisticRegression(solver='liblinear',
            penalty='l2', fit_intercept=fit_intercept, C=Cw, intercept_scaling=np.sqrt(Cb/Cw))
        self.fit_intercept = fit_intercept

        if self.fit_intercept:
            self.mean = self.full_weights()
        else:
            self.mean = self.lr_weight

        self.mean_prior = self.mean.copy()
        self.invcov_prior = self.invcov.copy()

        if posttype == 'Laplace':
            self.post_update = self._laplace_update
        elif posttype == 'Variational':
            self.post_update = self._variational_update
        else:
            raise ValueError('Enter valid posterior approximation!')

        if X is not None:

            assert(Yvec.shape[0] == X.shape[0])

            # train model
            self.trainLR()

            # perform posterior update
            self.post_update(None, None, tol)

    def add_point(self, x, y):
        """
        Inputs:
            x: d-length numpy vector encoding data point
            y: {0,1} class label
        Outputs:
            LR_int: time interval for LR training
            post_int: time interval for posterior update
        """

        # add point to training data
        if self.X is None:
            self.X = np.expand_dims(x,0)
            self.Yvec = np.array([y])
        else:
            self.Yvec = np.append(self.Yvec,y)
            self.X = np.vstack((self.X,x))

        start = time.time()

        # retrain
        self.trainLR()
        LRtime = time.time()

        # perform posterior update
        self.post_update(None, None, self.tol)
        post_time = time.time()

        LR_int = LRtime - start
        post_int = post_time - LRtime

        return LR_int, post_int

    def full_weights(self):
        """
        Returns weights, bias (0 if fit_intercept=False)
        """

        if self.fit_intercept:
            return np.append(self.lr_weight,self.lr_bias)
        else:
            return np.append(self.lr_weight,0)

    def get_test_acc(self, Xtest, Ytest):
        """
        Returns accuracy with respect to ground-truth dataset Xtest with labels Ytest
        """

        if self.fit_intercept:
            Xtest_ext = np.hstack((Xtest,-np.ones((Xtest.shape[0],1))))
            Xtest_proj = Xtest_ext.dot(self.full_weights())
        else:
            Xtest_proj = Xtest.dot(self.lr_weight)

        Yclass = (Xtest_proj > 0).astype(int)
        acc = np.mean(Yclass == Ytest)

        return acc

    def get_train_acc(self):
        """
        Returns training accuracy
        """

        return self.get_test_acc(self.X,self.Yvec)

    def get_classifier(self):
        """
        Returns classifier weight and bias
        """
        return self.lr_weight, self.lr_bias

    def post_stats(self):
        """
        Returns posterior mean and covariance
        """
        return self.mean, np.linalg.inv(self.invcov)

    def sample_posterior(self, Nsamp=30, full=False):
        """
        Returns Nsamp samples of posterior. full flag pads with zeros if fit_intercept=False
        """

        post_samp = np.random.multivariate_normal(self.mean,
                                             np.linalg.inv(self.invcov), (Nsamp,))

        if full and not self.fit_intercept:
            post_samp = np.hstack((post_samp,np.zeros((Nsamp,1))))

        return post_samp

    def plot(self, han=389, Nhyp_sample=50, stylized=False, plotmean=True, ploteig=True, Ypool=None):
        """
        Debug plotting of model
        Inputs:
            han: figure handle
            Nhyp_sample: number of posterior samples
            stylized: flag for figure stylization
            plotmean: flag to plot posterior mean
            ploteig: flag to plot maximum eigenvector
            Ypool: ground-truth labels for plotting
        """

        if stylized and self.d != 2:
            raise ValueError('Stylized plotting not compatible with d != 2')

        c0 = 'r' # class 0 color
        c1 = 'b' # class 1 color

        if self.X is not None and self.Yvec is not None:
            idx0 = [ii for ii in range(len(self.Yvec)) if self.Yvec[ii]==0]
            idx1 = [ii for ii in range(len(self.Yvec)) if self.Yvec[ii]==1]

        full_weights = self.full_weights()
        hyp_sample= self.sample_posterior(Nhyp_sample, full=True)

        if stylized:

            fig = plt.figure(han)
            plt.clf()

            x0range = np.array([-3,3])
            x1range = np.array([-3,3])

            if self.Xpool is not None:

                if Ypool is not None:
                    ec = [c0 if y==0 else c1 for y in Ypool]
                    plt.scatter(self.Xpool[:,0],self.Xpool[:,1], facecolors='None', edgecolors=ec, alpha=0.25, linewidths=1.5)

                else:
                    plt.scatter(self.Xpool[:,0],self.Xpool[:,1], facecolors='None', edgecolors='k')

                Xmin = np.min(self.Xpool,axis=0)
                Xmax = np.max(self.Xpool,axis=0)
            else:
                Xmin = np.min(self.X,axis=0)
                Xmax = np.max(self.X,axis=0)

            x0range = np.array([Xmin[0],Xmax[0]])
            x1range = np.array([Xmin[1],Xmax[1]])

            if self.X is not None and self.Yvec is not None:
                plt.scatter(self.X[idx0,0], self.X[idx0,1], c=c0, edgecolors='k', linewidths=1.5)
                plt.scatter(self.X[idx1,0], self.X[idx1,1], c=c1, edgecolors='k', linewidths=1.5)

            Wmean, Wcov = self.post_stats()
            eigvals,eigvecs = np.linalg.eig(Wcov)
            max_idx = np.argmax(np.abs(eigvals))
            max_vec = eigvecs[:,max_idx]

            if plotmean:
                w = Wmean
                wmag = np.linalg.norm(w)
                x_base = (full_weights[-1]/wmag**2)*w
                scale = 1
                plt.arrow(x_base[0], x_base[1], scale*w[0]/wmag, scale*w[1]/wmag, width=0.03, length_includes_head=True, color='c')
            if ploteig:
                w = max_vec
                wmag = np.linalg.norm(w)
                x_base = (full_weights[-1]/wmag**2)*w
                scale = 1
                plt.arrow(x_base[0], x_base[1], scale*w[0]/wmag, scale*w[1]/wmag, width=0.03, length_includes_head=True, color='m')

            plt.plot(x0range,(-full_weights[0]*x0range + full_weights[-1])/full_weights[1], c='k',linewidth=3)

            w = full_weights[0:2]
            wmag = np.linalg.norm(w)
            x_base = (full_weights[-1]/wmag**2)*w
            scale = 1
            plt.arrow(x_base[0], x_base[1], scale*w[0]/wmag, scale*w[1]/wmag, width=0.03, length_includes_head=True, color='k')

            for h in hyp_sample:
                plt.plot(x0range,(-h[0]*x0range + h[2])/h[1], c='k', alpha=0.1, linewidth=1.5)

            ax = plt.gca()

            ax.set_xlim(x0range)
            ax.set_ylim(x1range)
            ax.set_aspect('equal', adjustable='box')

            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)


        else:

            if self.d <= 2:
                fig = plt.figure(han)
                plt.clf()

                if self.d == 1:

                    ax1 = fig.add_subplot(121)

                    if self.Xpool is not None:
                        plt.scatter(self.Xpool[:,0], np.zeros(self.Xpool.shape[0]), facecolors='None', edgecolors='k')

                    if self.X is not None and self.Yvec is not None:
                        plt.scatter(self.X[idx0,0], np.zeros(len(idx0)), c=c0)
                        plt.scatter(self.X[idx1,0], np.zeros(len(idx1)), c=c1)

                    plt.plot([full_weights[-1]/full_weights[:-1],full_weights[-1]/full_weights[:-1]],[-1,1],c='k',linewidth=2)

                    for h in hyp_sample:
                        plt.plot([h[1]/h[0],h[1]/h[0]],[-1,1],c='k', alpha=0.2)

                    ax2 = fig.add_subplot(122)

                    plt.scatter(hyp_sample[:,0],hyp_sample[:,1])

                elif self.d == 2:

                    x0range = np.array([-3,3])
                    x1range = np.array([-3,3])

                    ax1 = fig.add_subplot(121)

                    if self.Xpool is not None:
                        plt.scatter(self.Xpool[:,0],self.Xpool[:,1], facecolors='None', edgecolors='k')

                        Xmin = np.min(self.Xpool,axis=0)
                        Xmax = np.max(self.Xpool,axis=0)
                    else:
                        Xmin = np.min(self.X,axis=0)
                        Xmax = np.max(self.X,axis=0)

                    x0range = np.array([Xmin[0],Xmax[0]])
                    x1range = np.array([Xmin[1],Xmax[1]])

                    if self.X is not None and self.Yvec is not None:
                        plt.scatter(self.X[idx0,0], self.X[idx0,1], c=c0)
                        plt.scatter(self.X[idx1,0], self.X[idx1,1], c=c1)

                    plt.plot(x0range,(-full_weights[0]*x0range + full_weights[-1])/full_weights[1], c='k',linewidth=2)

                    for h in hyp_sample:
                        plt.plot(x0range,(-h[0]*x0range + h[2])/h[1], c='k', alpha=0.2)

                    ax1.set_xlim(x0range)
                    ax1.set_ylim(x1range)
                    ax1.set_aspect('equal', adjustable='box')

                    ax2 = fig.add_subplot(1,2,2, projection='3d')
                    ax2.scatter(hyp_sample[:,0],hyp_sample[:,1],hyp_sample[:,2])
                    lim3d = 3*np.sqrt(self.Cw)

                    ax2.axes.set_xlim3d(left=-lim3d, right=lim3d)
                    ax2.axes.set_ylim3d(bottom=-lim3d, top=lim3d)
                    ax2.axes.set_zlim3d(bottom=-lim3d, top=lim3d)

    def utilmap(self, utilfun, han=400):
        """
        plots map of utility function

        utilfun: utility function. assumes takes in dx1 vector
        han (optional): figure handle
        """

        if self.d != 2:
            raise ValueError('Not compatible with d != 2')

        plt.figure(han)
        plt.clf()

        if self.Xpool is not None:
            Xmin = np.min(self.Xpool,axis=0)
            Xmax = np.max(self.Xpool,axis=0)
            x0range = np.array([Xmin[0],Xmax[0]])
            x1range = np.array([Xmin[1],Xmax[1]])
        else:
            x0range = np.array([-3,3])
            x1range = np.array([-3,3])

        res = 0.01
        x0_vec = np.arange(x0range[0],x0range[1],res)
        x1_vec = np.arange(x1range[0],x1range[1],res)
        X0,X1 = np.meshgrid(x0_vec,x1_vec)
        Util = np.zeros(X0.shape)

        for r in range(Util.shape[0]):
            for c in range(Util.shape[1]):
                 Util[r,c],_ = utilfun(np.array([X0[r,c],X1[r,c]]))

        plt.imshow(X=Util, cmap='viridis', origin='lower')

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    def trainLR(self):
        """
        Trains logistic regression classifier
        """

        if self.X is not None and self.Yvec is not None and len(np.unique(self.Yvec)) == 2:
            self.lr.fit(self.X,self.Yvec)
            self.lr_weight = self.lr.coef_[0,:]

            if self.fit_intercept:
                self.lr_bias = -self.lr.intercept_
            else:
                self.lr_bias = None

    def _laplace_update(self, Xinput=None, Yinput=None, tol=None, maxiter=None, debug_print=False):
        """
        Computes Laplace approximation for examples Xinput with labels Yinput (see Bishop 2006)

        Inputs:
            Xinput: (Npoints,d) array of examples
            Yinput: (Npoints,) vector labels (0 or 1)
            tol: tolerance for EM convergence (ignore for Laplace update)
            maxiter: maximum number of EM iterations (ignore for Laplace update)
            debug_print: flag for debug printing
        """

        if not Xinput:
            X = self.X
        else:
            X = Xinput

        # format data
        if self.fit_intercept:
            if X.ndim == 1:
                Xexp = np.expand_dims(np.append(X,-1),0)
            else:
                Xexp = np.hstack((X,-np.ones((X.shape[0],1))))
        elif X.ndim == 1:
            Xexp = np.expand_dims(X,0)
        else:
            Xexp = X

        self.invcov = self.invcov_prior.copy()
        for ii in range(Xexp.shape[0]):

            x = Xexp[np.newaxis,ii,:].T

            if self.fit_intercept:
                likelihood_1 = self.likelihood(x[:-1,0], 1)
            else:
                likelihood_1 = self.likelihood(x[:,0], 1)

            self.invcov = self.invcov + likelihood_1*(1-likelihood_1)*x.dot(x.T)

        if self.fit_intercept:
            self.mean = self.full_weights()
        else:
            self.mean = self.lr_weight

        return True

    def _variational_update(self, Xinput, Yinput, tol=1e-6, maxiter=100000, debug_print=False):
        """
        Computes update for variational approximation in (Jaakkola & Jordan 2000)
            after receiving labeled example (x,y)

        Inputs:
            X: (Npoints,d) array of examples
            Y: (Npoints,) vector labels (0 or 1)
            tol: tolerance for EM convergence
            maxiter: maximum number of EM iterations
            debug_print: flag for debug printing
        """

        if not Xinput:
            X = self.X
            Y = self.Yvec
        else:
            X = Xinput
            Y = Yinput

        # format data
        if self.fit_intercept:
            if X.ndim == 1:
                Xexp = np.expand_dims(np.append(X,-1),0)
            else:
                Xexp = np.hstack((X,-np.ones((X.shape[0],1))))
        elif X.ndim == 1:
            Xexp = np.expand_dims(X,0)
        else:
            Xexp = X

        lambda_ = lambda eps : np.tanh(eps/2) / (4*eps)

        mean_pos = self.mean
        invcov_pos = self.invcov
        rel_change = 2*tol

        eps = np.zeros(len(Xexp)) + 1e-10
        eps_new = np.zeros(len(Xexp)) + 1e-10

        steps_taken = 0
        order = 0

        while rel_change > tol and steps_taken < maxiter:

            invcov_pos_new = self.invcov_prior.copy()
            mean_pos_new = self.invcov_prior.dot(self.mean_prior)

            for ii in range(Xexp.shape[0]):

                x = Xexp[np.newaxis,ii,:].T
                y = Y[ii]

                eps_new[ii] = np.sqrt(np.asscalar(x.T.dot(np.linalg.inv(invcov_pos)).dot(x))
                    + (x[:,0].dot(mean_pos))**2)

                invcov_pos_new = invcov_pos_new + 2*lambda_(eps_new[ii])*x.dot(x.T)
                mean_pos_new = mean_pos_new + (y-1/2)*x[:,0]

            rel_change = np.linalg.norm(eps_new - eps,2) / np.linalg.norm(eps,2)

            steps_taken += 1

            if debug_print and (steps_taken % 10**order == 0):
                print('Steps: {}    relative change: {:.4e}'.format(steps_taken, rel_change))

                if steps_taken % 10**(order+1) == 0:
                    order += 1

            eps = eps_new.copy()

            mean_pos_new = np.linalg.inv(invcov_pos_new).dot(mean_pos_new)

            invcov_pos = invcov_pos_new
            mean_pos = mean_pos_new

        self.invcov = invcov_pos
        self.mean = mean_pos

        return True

    def likelihood(self, x, y, u=None):
        """
        Inputs:
            x: d-length numpy vector encoding data point
            y: {0,1} class label
            u: d+1 length vector, or Nsamp x (d+1) array
        Outputs:
            response likelihood 1/(1+exp(-u^T [x;-1]))
        """

        assert(x.ndim) == 1
        assert(len(x)) == self.d

        if u is None:
            u = self.full_weights()
        elif u.ndim==1:
            assert(u.shape==(self.d+1,))
        else:
            assert(u.shape[1]==self.d+1)

        tol = 1e-14 # constant floor and ceiling, for numerical stability
        p1 = np.minimum(np.maximum(1/(1+np.exp(-np.dot(u,np.append(x,-1)))), tol), 1-tol)
        return y*p1 + (1-y)*(1-p1)

if __name__ == '__main__':

    # working example

    # generate data
    datagen = Datagen(DatasetType.CROSS, classlist0=None, classlist1=None)
    X,Y,_,_,_,_ = datagen.genData(100, preprocess=True)

    # instantiate LR object
    posttype = 'Variational'; fig = 1
    LRobj = BayesLR(X=X, Yvec=Y, Cw=100, fit_intercept=False, posttype=posttype)

    # plot trained classifier
    LRobj.plot(han=fig, Nhyp_sample=50, stylized=True)
    plt.show()

    # print train accuracy
    print('Train accuracy: {}'.format(LRobj.get_train_acc()))
