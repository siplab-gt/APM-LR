# -*- coding: utf-8 -*-

DESCRIPTION = """
Experimental framework for APM testing.
"""

# specify single-thread processing
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import warnings
import argparse
import pickle
from BayesLR import BayesLR
from LR_active_selectors import Utility, select_example, get_utilfun
from datasets import Datagen, DatasetType

# set up arg parser
parser = argparse.ArgumentParser(DESCRIPTION)
parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no seed)')
parser.add_argument('--nfull', type=int, default=600, help='size of full dataset')
parser.add_argument('--ntrials', type=int, default=1, help='number of trials per experiment')
parser.add_argument('--nqueries', type=int, default=50, help='number of queries per run (-1 for full dataset)')
parser.add_argument('--preprocess', action='store_true', help='enables data preprocessing')
parser.add_argument('--fit_intercept', action='store_true', help='enables hyperplane intercept (EXPERIMENTAL)')
parser.add_argument('--doplot', action='store_true', help='enables plotting')
parser.add_argument('--verbose', action='store_true', help='enables debug printing')
parser.add_argument('--query_verbose', action='store_true', help='enables debug printing at query level (needs verbose to also be enabled)')
parser.add_argument('--savedata', action='store_true', help='enables data saving')
parser.add_argument('--savefolder', type=str, default='./', help='folder to save data in')
parser.add_argument('--savefile', type=str, default='test', help='file name (excluding .mat)')
parser.add_argument('--methods', type=str, nargs='+', default=['INFOGAIN'],
                    choices=['INFOGAIN', 'UNCERTAINTY', 'RANDOM', 'APMLR', 'BALD', 'MAXVAR'])
parser.add_argument('--methods_type', type=int, nargs='+', default=[0], help='list of indices to specify type of each method')
parser.add_argument('--methods_plot', type=int, nargs='+', default=[0], help='list of plotting flags (1 for plotting in method)')
parser.add_argument('--dataset_type', type=str, default='CLOUDS',
                    choices=['CROSS', 'HORSESHOE', 'CLOUDS', 'VEHICLE', 'LETTER', 'AUSTRA', 'WDBC'])
parser.add_argument('--classlist0', type=str, nargs='+', default=None, help='list of classes composing superclass 0')
parser.add_argument('--classlist1', type=str, nargs='+', default=None, help='list of classes composing superclass 1')
parser.add_argument('--burnin', type=int, default=0, help='Number of random burn-in queries')
parser.add_argument('--nseed', type=int, default=1, help='Number of seeds per class')
parser.add_argument('--plotpause', type=float, default=1.0, help='plot pause length (s)')
parser.add_argument('--posttype', type=str, default='Variational', help='Posterior approximation type {Laplace, Variational}',
                    choices=['Variational','Laplace'])
parser.add_argument('--doanalysis', action='store_true', help='enables optional data analysis')
parser.add_argument('--locksphere', action='store_true', help='normalizes data to unit sphere')
parser.add_argument('--Cw', type=float, default=100.0, help="""Coordinatewise variance on isotropic Gaussian weights prior. Equivalent to
               inverse of l2 penalty""")
opt = parser.parse_args()
print(opt)

# disable LR bias
if opt.fit_intercept:
    raise ValueError('Fit intercept not yet supported.')

# methods names, plotting, and types must match in length
if any(len(x) != len(opt.methods) for x in [opt.methods_type, opt.methods_plot]):
    raise ValueError('Length of methods_type, methods_plot, and methods must match')

# unpack parser
methods = [Utility[m] for m in opt.methods]
dataset_type = DatasetType[opt.dataset_type]
preprocess = opt.preprocess
nqueries = opt.nqueries
verbose = opt.verbose
query_verbose = opt.query_verbose

# generate temporary dataset to assess size
datagen = Datagen(dataset_type=dataset_type, classlist0=opt.classlist0, classlist1=opt.classlist1)
_,_,_,_,train_idx_TEMP,test_idx_TEMP = datagen.genData(opt.nfull, preprocess)

# ask training set size, minus 'nseed' seed points per class
if nqueries < 0:
    nqueries = len(train_idx_TEMP) - 2*opt.nseed

if opt.seed >= 0:
    np.random.seed(opt.seed)
    warnings.warn('Setting numpy seed!')

# initialize results
test_acc = np.zeros((len(methods), opt.ntrials, nqueries))
selection_time = np.zeros((len(methods), opt.ntrials, nqueries))
LR_time = np.zeros((len(methods), opt.ntrials, nqueries))
post_time = np.zeros((len(methods), opt.ntrials, nqueries))
selected_train_idx = np.zeros(shape=(len(methods), opt.ntrials, nqueries), dtype=np.int32)
           
train_idx_all = -1*np.ones(shape=(opt.ntrials, len(train_idx_TEMP)), dtype=np.int32)
test_idx_all = -1*np.ones(shape=(opt.ntrials, len(test_idx_TEMP)), dtype=np.int32)
 
if opt.doanalysis:
    ent_all = np.zeros((len(methods), opt.ntrials, nqueries))
    cond_ent_all = np.zeros((len(methods), opt.ntrials, nqueries))
    sel_mean_all = np.zeros((len(methods), opt.ntrials, nqueries))
    sel_var_all = np.zeros((len(methods), opt.ntrials, nqueries))
    max_eig_all = np.zeros((len(methods), opt.ntrials, nqueries))
    min_eig_all = np.zeros((len(methods), opt.ntrials, nqueries))
    det_cov_all = np.zeros((len(methods), opt.ntrials, nqueries))
    tr_cov_all = np.zeros((len(methods), opt.ntrials, nqueries))
    
    LRobj_dummy = BayesLR(X=None, Yvec=None, d=datagen.d, fit_intercept=opt.fit_intercept,
                        Xpool=None, posttype=opt.posttype)
    wmean_dummy, wcov_dummy = LRobj_dummy.post_stats()
    weights_full_dummy = LRobj_dummy.full_weights()
    
    wcov_all = np.zeros((len(methods), opt.ntrials, nqueries, *wcov_dummy.shape))
    wmean_all = np.zeros((len(methods), opt.ntrials, nqueries, len(wmean_dummy)))
    west_all = np.zeros((len(methods), opt.ntrials, nqueries, len(weights_full_dummy)))
    
else:
    ent_all = []
    cond_ent_all = []
    sel_mean_all = []
    sel_var_all = []
    max_eig_all = []
    min_eig_all = []
    det_cov_all = []
    tr_cov_all = []
    
    wcov_all = []
    wmean_all = []
    west_all = []
    
# save plot line names
plot_names = ['EMPTY']*len(methods)

plt.close('all')

# iterate over trials
for ti in range(opt.ntrials):

    if verbose:
        print('    Trial: {} / {}'.format(ti+1,opt.ntrials))
    
    # generate data
    Xpool,Ypool,Xtest,Ytest,train_idx,test_idx = datagen.genData(opt.nfull, preprocess)
    
    train_idx_all[ti] = train_idx
    test_idx_all[ti] = test_idx
    
    if opt.locksphere:
        Xpool_norm = np.linalg.norm(Xpool, axis=1)
        Xpool = Xpool / Xpool_norm[:,np.newaxis]
        Xtest_norm = np.linalg.norm(Xtest, axis=1)
        Xtest = Xtest / Xtest_norm[:,np.newaxis]
    
    # calculate dataset energy
    B = np.linalg.norm(Xpool, axis=1).max()
    
    if verbose:
        print('Maximum example norm: {}'.format(B))
    
    # debug plotting, view training pool
    if opt.doplot:
        fig = plt.figure()
        plt.clf()
        datagen.debugPlot(Xpool,Ypool,plt.gca())
        plt.pause(3)
        plt.savefig(fname='dataset.pdf',format='pdf', bbox_inches='tight')
        
    # initialize empty training set
    X = None
    Yvec = None
    
    # initialize full pool
    pool_idx_start = list(range(len(Xpool)))
    
    # seed each class
    if opt.nseed > 0:
        classes = np.unique(Ypool)
        idx0 = [ii for ii in range(len(Ypool)) if Ypool[ii]==classes[0]]
        idx1 = [ii for ii in range(len(Ypool)) if Ypool[ii]==classes[1]]
        
        # shuffle indices to select random examples
        np.random.shuffle(idx0)
        np.random.shuffle(idx1)
        
        # select random seeds of each class
        idx_seed = idx0[0:opt.nseed] + idx1[0:opt.nseed]
        
        # add to training data
        X = Xpool[idx_seed]
        Yvec = Ypool[idx_seed]
        
        # remove seeded indices from pool
        for ii in idx_seed:
            pool_idx_start.remove(ii)
    
    # cycle through each method. all methods share same trial datasets, seed points
    for mi in range(len(methods)):
        this_method = methods[mi]
        this_method_type = opt.methods_type[mi]
        this_method_plot = opt.methods_plot[mi]
        
        if verbose:
            print('Method: {}'.format(this_method.name))

        # initialize classifier
        LRobj = BayesLR(X=X, Yvec=Yvec, d=datagen.d, fit_intercept=opt.fit_intercept,
                        Xpool=Xpool, posttype=opt.posttype, Cw=opt.Cw)
            
        # initialize index set, which will be shortened as more queries are asked
        pool_idx = pool_idx_start.copy()
        
        for qi in range(nqueries):
            
            if verbose and query_verbose:
                print('        Query {} / {}'.format(qi+1, nqueries))
        
            # pull posterior statistics
            Wmean, Wcov = LRobj.post_stats()
            metadata = {'Wmean':Wmean, 'Wcov':Wcov, 'B':B}
            
            plotfreq = 1 # plot frequency
            if opt.doplot and qi % plotfreq == 0:
                
                if this_method != Utility.RANDOM:
                    utilfun = get_utilfun(this_method)
                    utilfun_x = lambda xv : utilfun(x=xv, LRobj=LRobj, method_type=this_method_type, metadata=metadata, doplot=False)
                else:
                    utilfun_x = lambda xv : (np.random.rand(), np.random.rand())
                
                h = plt.figure()
                LRobj.utilmap(utilfun=utilfun_x, han=h.number)
                plt.pause(opt.plotpause)
                plt.savefig(fname= this_method.name + str(qi) + 'asked_util.pdf',format='pdf', bbox_inches='tight')
                
                h = plt.figure()
                LRobj.plot(han=h.number, Nhyp_sample=100, stylized=True, plotmean=(this_method==Utility.APMLR), ploteig=(this_method==Utility.APMLR), Ypool=Ypool)
                plt.pause(opt.plotpause)
                plt.savefig(fname= this_method.name + str(qi) + 'asked.pdf',format='pdf', bbox_inches='tight')
                
                plt.close('all')
            
            # track timing
            start = time.time()
            
            # burnin period, of random querying
            if qi < opt.burnin:
                if verbose and query_verbose:
                    print('        BURNIN on query {}, out of {} burnin queries'.format(qi, opt.burnin))
                    
                idx, util_val, util_options = select_example(X=Xpool[pool_idx,:], LRobj=LRobj, utility=Utility.RANDOM, method_type=0)
            else:
                
                # only calculate maximum eigenvalue for APMLR
                if this_method == Utility.APMLR:
                    eigvals,_ = np.linalg.eig(Wcov)
                    maxeig = np.max(np.abs(eigvals))
                    metadata['maxeig'] = maxeig
            
                # select active query
                idx, util_val, util_options = select_example(X=Xpool[pool_idx,:], 
                    LRobj=LRobj, utility=this_method, method_type=this_method_type, 
                    metadata=metadata, doplot=this_method_plot)

            # selection time tracking
            this_selection_time = time.time() - start
            
            # select index, delete from pool
            Xidx = pool_idx[idx]
            del pool_idx[idx]
            
            selected_train_idx[mi, ti, qi] = Xidx
            
            # compute optional analyses, BEFORE updating posterior
            if opt.doanalysis:
                
                eigvals,_ = np.linalg.eig(Wcov)
                maxeig = np.max(np.abs(eigvals))
                metadata['maxeig'] = maxeig
                    
                apmlr = get_utilfun(Utility.APMLR)
                _, apm_opts = apmlr(Xpool[Xidx], LRobj, 0, metadata, doplot=False)
                
                sel_mean_all[mi, ti, qi] = apm_opts['EL']
                sel_var_all[mi, ti, qi] = apm_opts['VL']
                                    
                max_eig_all[mi, ti, qi] = np.max(eigvals)
                min_eig_all[mi, ti, qi] = np.min(eigvals)
                det_cov_all[mi, ti, qi] = np.linalg.det(Wcov)
                tr_cov_all[mi, ti, qi] = np.trace(Wcov)
                
                infogain = get_utilfun(Utility.INFOGAIN)
                _, info_options = infogain(Xpool[Xidx], LRobj, 0, metadata, doplot=False)
                
                ent_all[mi, ti, qi] = info_options['ent']
                cond_ent_all[mi, ti, qi] = info_options['cond_ent']
                
                wcov_all[mi, ti, qi] = Wcov
                wmean_all[mi, ti, qi] = Wmean
                west_all[mi, ti, qi] = LRobj.full_weights()

            
            # add to training set (updates classifier automatically)
            this_LR_int, this_post_int = LRobj.add_point(Xpool[Xidx], Ypool[Xidx])
            
            # get test accuracy
            this_train_acc = LRobj.get_train_acc()
            this_test_acc = LRobj.get_test_acc(Xtest,Ytest)
            
            test_acc[mi, ti, qi] = this_test_acc
            selection_time[mi, ti, qi] = this_selection_time
            LR_time[mi, ti, qi] = this_LR_int
            post_time[mi, ti, qi] = this_post_int
        
            if verbose and query_verbose:
                print('        Training accuracy: {}'.format(this_train_acc))
                print('        Test accuracy: {}'.format(this_test_acc))
                print('        Selection time: {}'.format(this_selection_time))
                print('        LR time: {}'.format(this_LR_int))
                print('        Posterior time: {}'.format(this_post_int))
                print('')
                
            # save plot name, once
            if qi==nqueries-1 and ti==opt.ntrials-1:
                plot_names[mi] = util_options['plot_name']
                
    
    # accumulate data every trial
    if opt.savedata:
        if opt.classlist0 is None:
            strclass0 = ''
        else:
            strclass0 = '-'.join(opt.classlist0)

        if opt.classlist1 is None:
            strclass1 = ''
        else:
            strclass1 = '-'.join(opt.classlist1)
            
        fullsavepath = (opt.savefolder + opt.savefile + dataset_type.name + 
                    '_class0-' + strclass0 + '_class1-' + strclass1)
        
        savedict = {'test_acc':test_acc, 'selection_time':selection_time, 
                    'LR_time':LR_time, 'post_time':post_time,
                    'selected_train_idx':selected_train_idx,
                     'd':datagen.d, 'ntrials':opt.ntrials,
                     'nqueries':nqueries, 'methods':opt.methods,
                     'train_idx_all':train_idx_all, 'test_idx_all':test_idx_all,
                     'sel_mean_all':sel_mean_all, 'sel_var_all':sel_var_all,
                     'ent_all':ent_all, 'cond_ent_all':cond_ent_all,
                     'max_eig_all':max_eig_all, 'min_eig_all':min_eig_all,
                     'det_cov_all':det_cov_all, 'tr_cov_all':tr_cov_all,
                     'wcov_all':wcov_all, 'wmean_all':wmean_all, 'west_all':west_all}
                
        sio.savemat(fullsavepath + '.mat', savedict)
        with open(fullsavepath + '.pkl', 'wb') as handle:
            pickle.dump({'opt':opt, 'plot_names':plot_names}, handle)