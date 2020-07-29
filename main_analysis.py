# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:20:13 2020
based on paper by Oren Forkosh - https://www.nature.com/articles/s41593-019-0516-y
@author: Samoilov-Katz Yuval
"""
from utils.lda import LDA

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from math import pi

from sklearn.datasets import make_classification as mc
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn import preprocessing, manifold
import scipy.cluster.hierarchy as shc
from sklearn.manifold import TSNE
import ctypes
import warnings
warnings.filterwarnings('ignore')

def IDs(n,components):
    '''
    Generate a dataset of humans activity in "social box" (e.g. "ikea store")
    \ninput: n = number of subjects
    output: ICA df + saves figures in Results folder
    '''
    days = 4 #for each subject
    readouts = 60 # features per subject
    clusters = 5 # generate 5 unique classes, big 5 model of personality traits
    x,y = mc(n_samples=n,n_features=days*readouts, n_classes=clusters,n_clusters_per_class=1,n_informative=3,random_state=23)    
    
    #PCA - unsupervied, maximun variance on entire data
    fig, ax = plt.subplots(figsize=(16,12), nrows=3, ncols=1)    
    norm_data = preprocessing.scale(x)
    pca = PCA()
    pca.fit(norm_data)
    pca_data = pca.transform(norm_data)
    exp_var = np.round(pca.explained_variance_ratio_ *100, decimals=1)
    labels = ['PC' + str(n) for n in range(1,len(exp_var)+1)]       
    pca_df = pd.DataFrame(pca_data, columns=labels)
    ##visualize
    ax[0].bar(x=range(1,len(exp_var)+1),height=exp_var, tick_label=labels,color='k')
    ax[0].set_ylabel('Explained Variance(%)')
    ax[0].set_xlabel('PC')
    ax[0].set_title(f'PCA with {n} subjects') 
    ax2 = ax[0].twinx() 
    ax2.set_ylabel('Total Variance [%]')
    t_v = 0
    t = []
    for v in exp_var:
        t_v +=v
        t.append(t_v)
    ax2.plot(range(1,len(exp_var)+1), t, color='r')
    ax[1].scatter(pca_df.PC1,pca_df.PC2, facecolors='none', edgecolors='k')
    ax[1].set_xlabel(f'PC1 [Au]')
    ax[1].set_ylabel(f'PC2 [Au]') 
        
    ##kernel PCA - unsupervied, maximun variance on entire data
    kpca = KernelPCA(n_components = components, kernel='rbf')
    kpca.fit(norm_data)
    pca_data2 = kpca.transform(norm_data)
    labels2 = ['kPC' + str(n) for n in range(1,len(pca_data2[0,:])+1)]       
    pca_df2 = pd.DataFrame(pca_data2, columns=labels2)
    ##visualize
    ax[2].scatter(pca_df2.kPC1,pca_df2.kPC2, facecolors='none', edgecolors='k')
    ax[2].set_xlabel(f'PC1 [Au]')
    ax[2].set_ylabel(f'PC2 [Au]') 
    ax[2].set_title(f'Gaussian kernal PCA') 
    plt.tight_layout()
    plt.savefig(f'Results/{n}_PCA')
    plt.close()
    
    '''
    PCA sucks for this (or maybe he is right...)
    assumes (i) linear relationships (ii) Moments 1,2 are sufficient to desribe distribution of variabls
    even gaussian kernel didn't help - max variance is not a good measure to span the data on orthogonal cordinates  
    Let's try more complexed algos which Izhar introduced us: isomap, and t-SNE 
    '''
    
    #isomap - non linear expansion of MDS for multi D by geodezian distances between behaviors 
    iso = manifold.Isomap(n_neighbors=components, n_components=components)
    iso.fit(norm_data)
    manifold_2Da = iso.transform(norm_data)
    manifold_2D = pd.DataFrame(manifold_2Da, columns=['C1', 'C2'])
    manifold_2D['tags'] = y
    p = sns.color_palette("bright", len(set(y)))
    sns.scatterplot(x='C1', y='C2', data=manifold_2D, palette=p, hue='tags', legend='full')
    plt.xlabel(f'C1 [Au]')
    plt.ylabel(f'C2 [Au]') 
    plt.tight_layout()
    plt.savefig(f'Results/{n}_ISOMAP')
    plt.close()
    
    '''
    divides the data in lower yet complexed 2D shapes. 
    Yet with no exception, cannot catch the right clusters
    Now t-SNE analysis
    '''
    #t-SNE with tags for an optinal behavioral category
    m = TSNE(learning_rate=50)
    tsne_res = m.fit_transform(x)
    dft=pd.DataFrame(data=x)
    dft['x'] = tsne_res[:,0]
    dft['y'] = tsne_res[:,1]
    dft['tags'] = y
    plt.figure(figsize=(16,10))
    ##visualize
    sns.scatterplot(x="x", y="y", data=dft, palette=p, hue='tags', legend='full')
    plt.savefig(f'Results/{n}_tSNE')
    plt.close()
    
    '''
    again, no clear seperation between clusters.
    how can we deside? maybe truly a supervised method could help.
    lets try LDA, similar to the papers only now we implemented it manually in utils folder
    '''
    
    #LDA - fits to represents stable behavioral traits
    lda = LDA(components)
    lda.fit(norm_data,y)
    lda_data = lda.transform(norm_data)
    exp_var2 = np.round(lda.explained_var() *100, decimals=1)
    labels = ['LD' + str(n) for n in range(1,len(exp_var2)+1)]       
    lda_df = pd.DataFrame(lda_data, columns=labels)
    lda_df['tags'] = y
    
    fig, l = plt.subplots(figsize=(16,12), nrows=1, ncols=1)
    #draw a polygon of personality space from the IDs
    df_m = lda_df.groupby('tags').mean()
    df_m = df_m.sort_values(by=['LD2'])
    df_m = df_m.append(df_m.iloc[0,:])
    ##visualize
    sns.scatterplot(x='LD1', y='LD2', data=lda_df, palette=p, hue='tags', legend='full',ax=l)
    df_m.plot.scatter('LD1','LD2',c='k',s=380,ax=l)
    l.plot(df_m.LD1, df_m.LD2, 'k', zorder=1, lw=4)
    l.set_xlabel(f'LD1 [Au]')
    l.set_ylabel(f'LD2 [Au]') 
    plt.title('Average Personality Space\nLDA for behavioral repertoire')
    plt.tight_layout()
    plt.savefig(f'Results/{n}_LDA')
    plt.close()
    
    '''
    Cool!
    even though its random data with 5 classes, 
    LDA maneged to devide it to 5 but in larger datasets
    how about other supervised methods?
    Now we officialy move to cluster analysis
    '''
    
    # Hierarchical analysis to explore main clusters
    fig, axes = plt.subplots(figsize=(16,12), nrows=1, ncols=2)
    subjects = [str(n) for n in range(1,len(x)+1)]
    methods = ['complete','ward']
    exp = ['Maximum Distances','Minimium Variance']
    for i,m in enumerate(methods):
        ## visualize 
        dend = shc.dendrogram(shc.linkage(x, method=m),labels=subjects ,above_threshold_color='k', orientation='right',ax=axes[i])
        axes[i].title.set_text(f'{exp[i]} Hierarchy with {n} subjects')
        axes[i].set_ylabel('Subjects')
        axes[i].set_xlabel('Distance')      
    plt.savefig(f'Results/{n}_Hierarchical')
    plt.close()
    
    '''
    sample size affects the Buttom up hierarchical clustering.
    Minimum variance  - while small size gave us 6 clusters, normal size gave us 2 and large size gave us 4 clusters 
    maximun distance - noisy results
    Now, Let's try the notorious ICA
    '''
    #ICA for an optinal behavioral category
    m = FastICA(n_components=components)
    ica_res = m.fit_transform(x)
    dfi=pd.DataFrame(data=x)
    dfi['x'] = ica_res[:,0]
    dfi['y'] = ica_res[:,1]
    dfi['tags'] = y
    fig, a = plt.subplots(figsize=(16,12), nrows=1, ncols=1)
    ##visualize
    dfi = dfi.iloc[:,240:]
    #draw a polygon of personality space from the IDs
    df_m = dfi.groupby('tags').mean()
    df_m = df_m.sort_values(by=['y'])
    df_m = df_m.append(df_m.iloc[0,:])
    sns.scatterplot(x="x", y="y", data=dfi, palette=p, hue='tags', legend='full',ax=a)
    plt.title('Average Personality Space\nICA for behavioral repertoire')
    df_m.plot.scatter('x','y',c='k',s=380,ax=a)
    a.plot(df_m.x, df_m.y, 'k', zorder=1, lw=4)
    plt.savefig(f'Results/{n}_ICA')
    plt.close()
    
    '''
    Summary:
    the clusters emerging from the LDA and ICA analysis
    Among all sizes, a triad can reflect the data.
    it does not differ among these sample sizes, though much clearer in the largerst sample size
    Thereby, we represented the personality space on 2D, based on these three unique clusters, 
    even though there are really 5 clusters in this random data
    '''
    
ns = [336,672,840] # number of subjects
c = 2 #number of main components for dimensionality reduction
for n in ns:
    start = time.time()
    #ctypes.windll.user32.MessageBoxW(0, "loading analysis for {} subjects".format(n),"SDA final project" , 1)
    IDs(n,c)
    duration = time.time()-start
    print(f'----------------------\nnubmer of subjects: {n}\nTime elapsed: {duration:.2f} seconds')
print('----------------------\nDone! see Results folder')


