# -*- coding: utf-8 -*-

"""
APM analysis script
"""

import os
import pickle
import scipy.io as sio
import numpy as np
from datasets import Datagen, DatasetType
import matplotlib.pyplot as plt
import scipy.spatial
import warnings
import matplotlib

linewidth_global = 1.75

names_dict = {'INFOGAIN':['InfoGain'],
              'UNCERTAINTY':['Uncertainty'],
              'MAXVAR':['MaxVar'],
              'BALD':['BALD'],
              'RANDOM':['Random'],
              'APMLR':['APM-LR','APM-LR-U', 'APM-LR-V']}
                       

def save_comptime(metadata_dict, keys=[], maxtrials=-1, qlim=40):
    """
    Generates sample complexity plots
    
    Inputs:
        metadata: dictionary of dataset metadata
        keys: datasets to analyze. If None, analyze all keys
        maxtrials: maximum trials to load (-1 for all)
        qlim: sum computation times from queries [0,qlim)
    """

    # initialize
    if len(keys)==0:
        keys = metadata_dict.keys()

    selected_time_all = []
    LR_time_all = []
    post_time_all = []
    ntrials_all = []
    methods = []
    
    for key in keys:
        # load metadata
        metadata = metadata_dict[key]
        
        # load data
        data, pkl_data = load_from_meta(metadata, maxtrials)
        opt = pkl_data['opt']
        
        selected_time_qlim = np.sum(data['selection_time'][:,:,:qlim],2)
        LR_time_qlim = np.sum(data['LR_time'][:,:,:qlim],2)
        post_time_qlim = np.sum(data['post_time'][:,:,:qlim],2)
        nmethods = len(selected_time_qlim)
        
        if metadata['plotorder']:
            plotidx = metadata['plotorder']
        else:
            plotidx = list(range(nmethods))
            
        plot_names_orig = [names_dict[opt.methods[mi]][opt.methods_type[mi]] for mi in range(nmethods)]
        plot_names = [plot_names_orig[pi] for pi in plotidx]
        
        ntrials_all.append(selected_time_qlim.shape[1])
        selected_time_all.append(selected_time_qlim[plotidx])
        LR_time_all.append(LR_time_qlim[plotidx])
        post_time_all.append(post_time_qlim[plotidx])
        
        if methods == []:
            methods = plot_names
        elif methods != plot_names:
            raise ValueError('Mismatch in plot indices')
        
    selected_time_all = np.array(selected_time_all)
    LR_time_all = np.array(LR_time_all)
    post_time_all = np.array(post_time_all)
    common_time_all = LR_time_all + post_time_all
    total_time_all = selected_time_all + common_time_all
    
    selected_time_median = np.median(selected_time_all, axis=2)
    LR_time_median = np.median(LR_time_all, axis=2)
    post_time_median = np.median(post_time_all, axis=2)
    common_time_median = np.median(common_time_all, axis=2)
    total_time_median = np.median(total_time_all, axis=2)

    dataset_keys = list(keys)
    
    np.savetxt('ntrials.csv', ntrials_all, delimiter=',')
    np.savetxt('dataset_keys.csv', dataset_keys, delimiter=',', fmt='%s')
    np.savetxt('methods.csv', methods, delimiter=',', fmt='%s')
    
    np.savetxt('selected_time_median.csv', selected_time_median.T, delimiter=',')
    np.savetxt('LR_time_median.csv', LR_time_median.T, delimiter=',')
    np.savetxt('post_time_median.csv', post_time_median.T, delimiter=',')
    np.savetxt('common_time_median.csv', common_time_median.T, delimiter=',')
    np.savetxt('total_time_median.csv', total_time_median.T, delimiter=',')
    
    return selected_time_all, LR_time_all, post_time_all, methods
    

def samplecomp(metadata_dict, keys=[], maxtrials=-1, ploterr=0,
               saveplotpath=None, figsize=(8,5)):
    """
    Generates sample complexity plots
    
    Inputs:
        metadata: dictionary of dataset metadata
        keys: datasets to analyze. If None, analyze all keys
        maxtrials: maximum trials to load (-1 for all)
        ploterr: error type
            0: none
            1: standard deviation
            2: standard error
            3: 95-percentile
        saveplotpath: file path for saving figures, root. (None for no saving)
        figsize: figure size
        
    """
    
    percentile_alpha = 0.05
    
    # initialize
    if len(keys)==0:
        keys = metadata_dict.keys()
    
    for key in keys:
        
        # load metadata
        metadata = metadata_dict[key]
        
        if 'legend_flag' in metadata.keys():
            legend_flag = metadata['legend_flag']
        else:
            legend_flag = True
        
        # load data
        data, pkl_data = load_from_meta(metadata, maxtrials)
        test_acc = data['test_acc']
        nqueries = data['nqueries']
        ntrials = data['ntrials']
        opt = pkl_data['opt']
        
        if metadata['xr']:
            xr = metadata['xr']
        else:
            xr = [0,nqueries]
        
        # plotting
        testacc_fig = plt.figure()
        plt.clf()
        
        col = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        if metadata['plotorder']:
            plotidx = metadata['plotorder']
        else:
            plotidx = list(range(len(test_acc)))
        
        for mi,ci in zip(plotidx,list(range(len(col)))):
    
            line = np.squeeze(np.mean(test_acc[mi,:,:], axis=0))
            pline = plt.plot(list(range(nqueries)), line, color=col[ci], linewidth=linewidth_global)
            err_std = np.squeeze(np.std(test_acc[mi,:,:], axis=0))
            err_se = err_std/np.sqrt(ntrials)
            
            if ploterr == 1:
                err_upper = line + err_std
                err_lower = line - err_std
            elif ploterr == 2:
                err_upper = line + err_se
                err_lower = line - err_se
            elif ploterr == 3:
                err_upper = np.squeeze(np.percentile(a=test_acc[mi,:,:], q=100*(1-percentile_alpha/2), axis=0))
                err_lower = np.squeeze(np.percentile(a=test_acc[mi,:,:], q=100*percentile_alpha/2, axis=0))
                
                med = np.squeeze(np.median(test_acc[mi,:,:], axis=0))
                plt.plot(list(range(nqueries)), med, color=col[ci], linestyle='--')

            if ploterr > 0 :
                plt.fill_between(list(range(nqueries)), err_lower, err_upper, color=pline[0].get_color(), alpha=0.15)
                
            pline[0].set_label(names_dict[opt.methods[mi]][opt.methods_type[mi]])
    
        if saveplotpath:
            saveplotpath_key = saveplotpath + '_' + key
        else:
            saveplotpath_key = None
            
        if legend_flag:
            legend = {}
        else:
            legend = None
            
        make_figure(testacc_fig, xlabel='Query number', ylabel='Test accuracy', legend=legend,
                    figsize=figsize, saveplotpath=saveplotpath_key, xr=xr)
        
        print('Dataset: {}, # trials: {}, plotting in xrange: {}'.format(key,ntrials,xr))
        
        
def exploratory(metadata_dict, keys=[], maxtrials=-1, loadflag=0,
                datapath=None, saveplotpath=None, figsize=(8,5)):
    """
    Exploratory analysis
    
    Inputs:
        metadata: dictionary of dataset metadata
        keys: datasets to analyze. If None, analyze all keys
        maxtrials: maximum trials to load (-1 for all)
        loadflag: flag for save or load behavior
            0: no saving or loading
            1: save data to datapath
            2: load data from datapath
        datapath: path to saved data
        saveplotpath: file path for saving figures, root. (None for no saving)
        figsize: figure size
    """
    
    if len(keys)==0:
        keys = metadata_dict.keys()
        
    if loadflag == 1:
        data_saved = {}
        
    elif loadflag == 2:
        with open(datapath, 'rb') as f:
            data_loaded = pickle.load(f)
    
    #%% parameters
       
    for key in keys:
        
        metadata = metadata_dict[key]
        
        if loadflag == 1:
            data_saved_key = {}
        elif loadflag == 2:
            data_loaded_key = data_loaded[key]
        
        print('Analyzing ' + key + '...')
        gap = 1
        
        print('    '*gap + 'Load dataset...')
        
        # load data
        data, pkl_data = load_from_meta(metadata_dict[key], maxtrials)
        opt = pkl_data['opt']
        
        if opt.fit_intercept:
            warnings.warn('Fit intercept not supported in analysis!')
            
        dataset_type = DatasetType[opt.dataset_type]
        preprocess = opt.preprocess
        ntrials = data['ntrials']
        
        train_idx_all = data['train_idx_all']
        test_idx_all = data['test_idx_all']
        
        selected_train_idx = data['selected_train_idx']

        datagen = Datagen(dataset_type=dataset_type, classlist0=opt.classlist0, classlist1=opt.classlist1)

        nmethods = selected_train_idx.shape[0]
        ntrials = data['ntrials']
        nqueries = data['nqueries']
        d = data['d']
        
        if metadata['xr']:
            xr = metadata['xr']
        else:
            xr = [0,nqueries]
            
        if metadata['plotorder']:
            plotidx = metadata['plotorder']
        else:
            plotidx = list(range(nmethods))
        
        plot_names_orig = [names_dict[opt.methods[mi]][opt.methods_type[mi]] for mi in range(nmethods)]
        plot_names = [plot_names_orig[pi] for pi in plotidx]
        
        #%% dataset analysis
        
        print('    '*gap + 'Analyzing dataset...')
    
        # selected examples: exploration analysis
         
        if loadflag == 2:
            maxmin_dist = data_loaded_key['maxmin_dist']
            log_gram_det = data_loaded_key['log_gram_det']
        else:
            maxmin_dist = np.zeros((nmethods,ntrials,nqueries))
            log_gram_det = np.zeros((nmethods,ntrials,nqueries-d+1))
            
            for ti in range(ntrials):
                print('\r' + '    '*(gap) + 'Exploratory analysis...{:.0f}%'.format((ti+1)/ntrials*100),end='')
                
                train_idx = train_idx_all[ti]
                test_idx = test_idx_all[ti]
                
                # load data
                Xpool,Ypool,Xtest,Ytest,_,_ = datagen.genData(opt.nfull, preprocess, train_idx, test_idx)
                D = scipy.spatial.distance_matrix(Xpool,Xpool)
                
                selected_train_idx_ti = selected_train_idx[:,ti,:]
                Xpool_sel = Xpool[selected_train_idx_ti]
                
                seed_idx = [si for si in range(len(Xpool)) if si not in selected_train_idx_ti[0]]
                
                # maxmin analysis
                for mi in range(nmethods):
                    for qi in range(nqueries):
                        maxmin_dist[mi,ti,qi] = maxmin_explore(D,list(selected_train_idx_ti[mi,0:(qi+1)]) + seed_idx)
                            
                # Gram determinant analysis
                for mi in range(nmethods):
                    for qi in range(nqueries-d+1):
                        batch_idx = list(range(qi,qi+d))
                        _,log_gram_det[mi,ti,qi] = np.linalg.slogdet(np.matmul(Xpool_sel[mi,batch_idx],Xpool_sel[mi,batch_idx].T))
                
            print('')
            
            if loadflag == 1:
                data_saved_key['maxmin_dist'] = maxmin_dist
                data_saved_key['log_gram_det'] = log_gram_det
        
        mean_maxmin_dist = np.mean(maxmin_dist, axis=1)
        mean_log_gram_det = np.mean(log_gram_det, axis=1)
            
        # maximin distance
        maxmin_fig = plt.figure()
        plt.clf()
        plt.plot(mean_maxmin_dist[plotidx].T, linewidth=linewidth_global)
        
        if saveplotpath:
            saveplotpath_key = saveplotpath + '_' + key + '_maxmin'
        else:
            saveplotpath_key = None
            
        make_figure(maxmin_fig, xlabel='Query number', ylabel='Maximin distance', legend={'labels':plot_names, 'bbox_to_anchor':(1.04,0.5),'loc':'center left'},
                    figsize=figsize, saveplotpath=saveplotpath_key, xr=xr)
            
        # log determinant Gram
        gram_fig = plt.figure()
        plt.clf()
        plt.plot(mean_log_gram_det[plotidx].T, linewidth=linewidth_global)
        
        if saveplotpath:
            saveplotpath_key = saveplotpath + '_' + key + '_gramdet'
        else:
            saveplotpath_key = None
            
        if metadata['xr']:
            xr_gram = metadata['xr']
        else:
            xr_gram = [0,mean_log_gram_det.shape[1]]
            
        make_figure(gram_fig, xlabel='Window number', ylabel='Log determinant Gram', legend={'labels':plot_names, 'bbox_to_anchor':(1.04,0.5),'loc':'center left'},
                    figsize=figsize, saveplotpath=saveplotpath_key, xr=xr_gram)


        # selected examples: other analysis
        
        if opt.doanalysis:
            
            # marginal mean
            west_all = data['west_all']
           
            if loadflag == 2:
                hyp_dist_est = data_loaded_key['hyp_dist_est']
                
            else:
                # distance from hyperplane estimate
                hyp_dist_est = np.zeros(maxmin_dist.shape)
            
                for ti in range(ntrials):
                    print('\r' + '    '*(gap) + 'Full analysis...{:.0f}%'.format((ti+1)/ntrials*100),end='')
                    
                    train_idx = train_idx_all[ti]
                    test_idx = test_idx_all[ti]
                    
                    # load data
                    Xpool,Ypool,Xtest,Ytest,_,_ = datagen.genData(opt.nfull, preprocess, train_idx, test_idx)

                    # calculate distances to hyperplane
                    selected_train_idx_ti = selected_train_idx[:,ti]
                    Xpool_sel = Xpool[selected_train_idx_ti]
                    west_ti = west_all[:,ti]
                    
                    assert(Xpool_sel.shape[2] == west_ti.shape[2]-1)
                    hyp_dist_est[:,ti,:] = np.abs((np.sum(Xpool_sel*west_ti[:,:,:-1], axis=2) - 
                                            west_ti[:,:,-1]))/np.linalg.norm(west_ti[:,:,:-1],axis=2)

                print('')
                
                if loadflag == 1:
                    data_saved_key['hyp_dist_est'] = hyp_dist_est
            
            mean_hyp_dist_est= np.mean(hyp_dist_est,axis=1)
             
            # distance from hyperplane estimate
            uncertain_fig = plt.figure()
            plt.clf()
            plt.plot(mean_hyp_dist_est[plotidx].T, linewidth=linewidth_global)
            
            if saveplotpath:
                saveplotpath_key = saveplotpath + '_' + key + '_dist-hypest'
            else:
                saveplotpath_key = None
                
            make_figure(uncertain_fig, xlabel='Query number', ylabel='Distance to estimated hyperplane',
                        legend={'labels':plot_names, 'bbox_to_anchor':(1.04,0.5),'loc':'center left'},
                        figsize=figsize, saveplotpath=saveplotpath_key, xr=xr, ylog=True)
            
        else:
            print('    '*(gap) + 'Skipping full analysis!')
            
        if loadflag == 1:
            data_saved[key] = data_saved_key
    
    print('Complete!')
    
    if loadflag == 1:
        with open(datapath, 'wb') as f:
            pickle.dump(data_saved, f)

#%%

def make_figure(han, xlabel='', ylabel='', legend=None, figsize=(8,5), saveplotpath=None, xr=None, yr=None, ylog=False):
    """
    Make figure
    
    Inputs:
        han: figure handle (object)
        xlabel: x label
        ylabel: y label
        legend: dictionary of legend arguments. If None, legend not called
        figsize: figure size
        saveplotpath: file path for saving figures, root. (None for no saving)
        xr: xlim range (default None)
        yr: ylim range (default None)
        ylog: if True, plot y-axis on log scale
    Outputs:
        han: figure handle
    """
    
    plt.figure(han.number)
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 16}) 

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)           
        
    if xr:
        plt.xlim(xr)
        
    if yr:
        plt.ylim(yr)
        
    if ylog:
        plt.yscale('log')
            
    if legend is not None:
        leg = plt.legend(**legend)
            
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.5)
    
    plt.grid(which='major',linewidth='0.5')
    han.set_size_inches(figsize)
    
    if saveplotpath:
        plt.savefig(fname=saveplotpath + '.pdf',format='pdf', bbox_inches='tight')
    
    return han
    
    
def load_from_meta(metadata, maxtrials=-1):
    """
    Loads and compiles datasets
    
    Inputs:
        metadata: dictionary of dataset metadata
        maxtrials: maximum trials to load (-1 for all)
    """
    
    data_all = []
    pkl_all = []
    
    for (root, nfile, suf) in zip(metadata['roots'], metadata['nfiles'], metadata['suffix']):
        for nf in range(nfile):
            path = os.path.join(root, metadata['expname'] + '_' + str(nf) + 
                                metadata['dataset'] + '_class0-' + metadata['class0'] + '_class1-' + 
                                metadata['class1'] + suf)
            
            data = sio.loadmat(path + '.mat', squeeze_me=False)
            
            with open(path + '.pkl', 'rb') as handle:
                pkl_data = pickle.load(handle)
                
            if len(data_all) == 0:
                data_all = data
                pkl_all = pkl_data
            else:
                for key in data.keys():
                    if key in ['test_acc','selection_time','LR_time','post_time',
                               'selected_train_idx',
                               'sel_mean_all','sel_var_all','ent_all','cond_ent_all',
                               'max_eig_all','min_eig_all','det_cov_all','tr_cov_all',
                               'wcov_all','wmean_all','west_all']:
                        
                        data_all[key] = np.concatenate((data_all[key], data[key]), axis=1)
                    
                    elif key in ['train_idx_all','test_idx_all']:
                        
                        data_all[key] = np.concatenate((data_all[key], data[key]), axis=0)
                    
                    elif key in ['d','methods','nqueries'] and any(data_all[key] != data[key]):
                        raise ValueError('Key ' + key + ' in ' + path + ' is not consistent with other files')
                    
                data_all['ntrials'] += data['ntrials']
                
    data_all['ntrials'] = data_all['ntrials'].item()
    data_all['nqueries'] = data_all['nqueries'].item()
    data_all['d'] = data_all['d'].item()
           
    if maxtrials > 0 and data_all['ntrials'] > maxtrials:
        data_all['ntrials'] = maxtrials
        
        for key in data_all.keys():
            if key in ['test_acc','selection_time','LR_time','post_time',
                                   'selected_train_idx',
                                   'sel_mean_all','sel_var_all','ent_all','cond_ent_all',
                                   'max_eig_all','min_eig_all','det_cov_all','tr_cov_all',
                                   'wcov_all','wmean_all','west_all']:
                data_all[key] = data_all[key][:,:maxtrials]
                
            elif key in ['train_idx_all','test_idx_all']:
                data_all[key] = data_all[key][:maxtrials]
    
    return data_all, pkl_all


def maxmin_explore(D,sel_idx):
    """
    Inputs:
        D: distance matrix of total pool
        sel_idx: index of selected (labeled) points
    Outputs:
        maxmin metric
    """
        
    unlab_idx = [xi for xi in range(len(D)) if xi not in sel_idx]
    
    if len(unlab_idx) > 0:
        D_unlab = D[unlab_idx]
        D_unlab_lab = D_unlab[:,sel_idx]
        
        return np.max(np.min(D_unlab_lab, axis=1), axis=0)
    else:
        return 0.0
    
    
#%%

def main():
    
    plt.close('all')
    
    globroot = os.path.join('..','results','fullrun')
    saveplotpath = os.path.join('.','fullrun')
    globnfiles = 10
    figsize = (8,5)
    
    keys_analysis = ['vehicle-cars']
    plotorder_main = [3,1,2,5,0,4]
    plotorder_analysis_short = [3,1,2,7,0,6]
    plotorder_analysis_long = [3,1,2,7,0,6,4,5]
        
    # global parameters
    maxtrials = 150
    qlim = 40
    
    # build metadata list
    metadata = {}
    
    # wdbc
    wdbc = {}
    wdbc['roots'] = [globroot]
    wdbc['nfiles'] = [globnfiles]
    wdbc['suffix'] = ['']
    wdbc['expname'] = 'fullrun_miscUCI'
    wdbc['dataset'] = 'WDBC'
    wdbc['class0'] = ''
    wdbc['class1'] = ''
    wdbc['xr'] = None
    wdbc['plotorder'] = plotorder_main
    
    # CLOUDS
    clouds = {}
    clouds['roots'] = [globroot]
    clouds['nfiles'] = [globnfiles]
    clouds['suffix'] = ['']
    clouds['expname'] = 'fullrun_toy'
    clouds['dataset'] = 'CLOUDS';
    clouds['class0'] = '';
    clouds['class1'] = '';
    clouds['xr'] = [0,75]
    clouds['plotorder'] = plotorder_main
    
    # HORSESHOE
    horseshoe = {}
    horseshoe['roots'] = [globroot]
    horseshoe['nfiles'] = [globnfiles]
    horseshoe['suffix'] = ['']
    horseshoe['expname'] = 'fullrun_toy'
    horseshoe['dataset'] = 'HORSESHOE';
    horseshoe['class0'] = '';
    horseshoe['class1'] = '';
    horseshoe['xr'] = [0,100]
    horseshoe['plotorder'] = plotorder_main
    
    # D,P
    dp = {}
    dp['roots'] = [globroot]
    dp['nfiles'] = [globnfiles]
    dp['suffix'] = ['']
    dp['expname'] = 'fullrun_letters'
    dp['dataset'] = 'LETTER';
    dp['class0'] = 'D';
    dp['class1'] = 'P';
    dp['xr'] = [0,125]
    dp['plotorder'] = plotorder_main
    
    # E,F
    ef = {}
    ef['roots'] = [globroot]
    ef['nfiles'] = [globnfiles]
    ef['suffix'] = ['']
    ef['expname'] = 'fullrun_letters'
    ef['dataset'] = 'LETTER';
    ef['class0'] = 'E';
    ef['class1'] = 'F';
    ef['xr'] = [0,125]
    ef['plotorder'] = plotorder_main
    
    # I,J
    ij = {}
    ij['roots'] = [globroot]
    ij['nfiles'] = [globnfiles]
    ij['suffix'] = ['']
    ij['expname'] = 'fullrun_letters'
    ij['dataset'] = 'LETTER';
    ij['class0'] = 'I';
    ij['class1'] = 'J';
    ij['xr'] = [0,200]
    ij['plotorder'] = plotorder_main
    
    # M,N
    mn = {}
    mn['roots'] = [globroot]
    mn['nfiles'] = [globnfiles]
    mn['suffix'] = ['']
    mn['expname'] = 'fullrun_letters'
    mn['dataset'] = 'LETTER';
    mn['class0'] = 'M';
    mn['class1'] = 'N';
    mn['xr'] = [0,125]
    mn['plotorder'] = plotorder_main
    
    # U,V
    uv = {}
    uv['roots'] = [globroot]
    uv['nfiles'] = [globnfiles]
    uv['suffix'] = ['']
    uv['expname'] = 'fullrun_letters'
    uv['dataset'] = 'LETTER';
    uv['class0'] = 'U';
    uv['class1'] = 'V';
    uv['xr'] = [0,125]
    uv['plotorder'] = plotorder_main
    
    # V,Y
    vy = {}
    vy['roots'] = [globroot]
    vy['nfiles'] = [globnfiles]
    vy['suffix'] = ['']
    vy['expname'] = 'fullrun_letters'
    vy['dataset'] = 'LETTER';
    vy['class0'] = 'V';
    vy['class1'] = 'Y';
    vy['xr'] = [0,200]
    vy['plotorder'] = plotorder_main
    
    # full
    vehicle_full = {}
    vehicle_full['roots'] = [globroot]
    vehicle_full['nfiles'] = [globnfiles]
    vehicle_full['suffix'] = ['']
    vehicle_full['expname'] = 'fullrun_miscUCI'
    vehicle_full['dataset'] = 'VEHICLE';
    vehicle_full['class0'] = 'saab-opel';
    vehicle_full['class1'] = 'bus-van';
    vehicle_full['xr'] = [0,150]
    vehicle_full['plotorder'] = plotorder_main
    
    # buses
    vehicle_bus = {}
    vehicle_bus['roots'] = [globroot]
    vehicle_bus['nfiles'] = [globnfiles]
    vehicle_bus['suffix'] = ['']
    vehicle_bus['expname'] = 'fullrun_miscUCI'
    vehicle_bus['dataset'] = 'VEHICLE';
    vehicle_bus['class0'] = 'bus';
    vehicle_bus['class1'] = 'van';
    vehicle_bus['xr'] = [0,150]
    vehicle_bus['plotorder'] = plotorder_main
    
    # austra
    austra = {}
    austra['roots'] = [globroot]
    austra['nfiles'] = [globnfiles]
    austra['suffix'] = ['']
    austra['expname'] = 'fullrun_miscUCI'
    austra['dataset'] = 'AUSTRA'
    austra['class0'] = ''
    austra['class1'] = ''
    austra['xr'] = [0,150]
    austra['plotorder'] = plotorder_main
    
    # CROSS
    cross = {}
    cross['roots'] = [globroot]
    cross['nfiles'] = [globnfiles]
    cross['suffix'] = ['']
    cross['expname'] = 'fullrun_toy'
    cross['dataset'] = 'CROSS';
    cross['class0'] = '';
    cross['class1'] = '';
    cross['xr'] = None
    cross['legend_flag'] = False
    cross['plotorder'] = plotorder_main
    
    # cars
    vehicle_cars = {}
    vehicle_cars['roots'] = [os.path.join('..','results','analysis')]
    vehicle_cars['nfiles'] = [50]
    vehicle_cars['suffix'] = ['']
    vehicle_cars['expname'] = 'fullanalysis'
    vehicle_cars['dataset'] = 'VEHICLE';
    vehicle_cars['class0'] = 'saab';
    vehicle_cars['class1'] = 'opel';
    vehicle_cars['xr'] = None
    vehicle_cars['plotorder'] = plotorder_analysis_short
    
    metadata['vehicle-full'] = vehicle_full
    metadata['vehicle-cars'] = vehicle_cars
    metadata['vehicle-transport'] = vehicle_bus  
    metadata['letterDP'] = dp
    metadata['letterEF'] = ef
    metadata['letterIJ'] = ij
    metadata['letterMN'] = mn
    metadata['letterUV'] = uv
    metadata['letterVY'] = vy
    metadata['austra'] = austra      
    metadata['wdbc'] = wdbc
    metadata['clouds'] = clouds
    metadata['cross'] = cross
    metadata['horseshoe'] = horseshoe
    
    # compute time
    save_comptime(metadata, keys=[], maxtrials=maxtrials, qlim=qlim)
    
    # sample complexity plots      
    samplecomp(metadata_dict=metadata,keys=[],maxtrials=maxtrials,ploterr=2,saveplotpath=saveplotpath,figsize=figsize)

    for key in keys_analysis:
        metadata[key]['plotorder'] = plotorder_analysis_long
        metadata[key]['xr'] = None
        
    # full trajectory plotting, with pseudo-APM methods
    samplecomp(metadata_dict=metadata,keys=keys_analysis,maxtrials=maxtrials,ploterr=0,saveplotpath=saveplotpath+'_all',figsize=figsize)            
               
    # exploratory analysis. loadflag=1 saves data, loadflag=2 loads data. loadflag=0 does not save or load
    exploratory(metadata,keys=keys_analysis,maxtrials=maxtrials,loadflag=1,datapath='./analyzed_data.pkl', saveplotpath=saveplotpath, figsize=figsize)     
     

if __name__ == '__main__':
    main()