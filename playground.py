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
import seaborn as sns

from sklearn.datasets import make_classification as mc
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn import preprocessing, manifold
import scipy.cluster.hierarchy as shc
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def main(n,components):
    '''
    Generate a dataset of humans activity in "social box" (e.g. "ikea store")
    \ninput: n = number of subjects
    \noutput: saves figures in Results folder
    '''
    days = 4 #for each subject
    readouts = 60 #for each subject
    clusters = 5
    #assuming there are 4 classes - IDs
    x,y = mc(n_samples=n,n_features=days*readouts, n_classes=clusters,n_clusters_per_class=1,n_informative=3,random_state=23)
    fig, ax = plt.subplots(figsize=(16,12), nrows=3, ncols=1)    
    
    #PCA - unsupervied, maximun variance on entire data
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
        
    #kernel PCA - unsupervied, maximun variance on entire data
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
    iso = manifold.Isomap(n_neighbors=5, n_components=components)
    iso.fit(norm_data)
    manifold_2Da = iso.transform(norm_data)
    manifold_2D = pd.DataFrame(manifold_2Da, columns=['C1', 'C2'])
    manifold_2D['tags'] = y

    sns.scatterplot(x='C1', y='C2', data=manifold_2D, palette=palette, hue='tags', legend='full')
    plt.xlabel(f'C1 [Au]')
    plt.ylabel(f'C2 [Au]') 
    plt.tight_layout()
    plt.savefig(f'Results/{n}_ISOMAP')
    plt.close()
    
    '''
    divides the data in lower interesting 2D shapes. 
    but with no exception, cannot catch the right clusters
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
    sns.scatterplot(x="x", y="y", data=dft, palette=palette, hue='tags', legend='full')
    plt.savefig(f'Results/{n}_tSNE')
    plt.close()
    
    '''
    again, no clear seperation between clusters...
    how can we deside? maybe a supervised method can help.
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
    
    ##visualize
    palette = sns.color_palette("bright", len(set(y)))
    sns.scatterplot(x='LD1', y='LD2', data=lda_df, palette=palette, hue='tags', legend='full')
    plt.xlabel(f'LD1 [Au]')
    plt.ylabel(f'LD2 [Au]') 
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Results/{n}_LDA')
    plt.close()
    
    '''
    even though its random data with 5 classes, 
    LDA maneged to devide it to 5 but with only 4 visually main classes which is consistent among sample sizes
    how about other supervised methods?
    we move to clustering.
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
    sample size affects Buttom up hierarchical clustering!
    Minimum variance  - while small sample size gave us 4 clusters, normal gave us 2 and large gave us 3 clusters (336,672,999)
    maximun distance - noisy results
    '''
    
    #ICA for an optinal behavioral category
    m = FastICA(n_components=components)
    ica_res = m.fit_transform(x)
    dfi=pd.DataFrame(data=x)
    dfi['x'] = ica_res[:,0]
    dfi['y'] = ica_res[:,1]
    dfi['tags'] = y
    plt.figure(figsize=(16,10))
    ##visualize
    sns.scatterplot(x="x", y="y", data=dfi, palette=palette, hue='tags', legend='full')
    plt.title('ICA a friend of dimensionality reduction')
    plt.savefig(f'Results/{n}_ICA')
    plt.close()
    
    '''
    the edges reflect the picture:
    the clusters emerging from the ICA analysis
    it does not differ among these sample sizes, though much clearer in the largerst sample size
    '''
    
ns = [336,672,999] # number of subjects
c = 2 #number of main components for dimensionality reduction
for n in ns:
    start = time.time()
    main(n,c)
    duration = time.time()-start
    print(f'----------------------\nnubmer of subjects: {n}\nTime elapsed: {duration:.2f} seconds')
print('----------------------\nDone! see Results folder')