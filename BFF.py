#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
from astropy import table
from astropy.io import ascii
import copy
import numpy as np
import matplotlib.pyplot as plt
import	multiprocessing as mp
#from plotsettings_py36 import *
from scipy import interpolate
import scipy
import scipy.optimize
from scipy.optimize import curve_fit
import time
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import statistics 
from extinction import ccm89, apply


# ##### Measuring how long it takes

# In[2]:


start = time.time()


# # (Super) Functions

# In[3]:


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


# In[4]:


A_v = 1.0

def Alam(lamin, A_v):

    #Add extinction with R_v= 3.1 and A_v = 1 
    flux = np.ones(len(lamin))
    redreturn = apply(ccm89(lamin, 1.0, 3.1), flux)
    
    return redreturn


# In[5]:


def select_templates(DATABASE, TYPES):

       
#    Selects templates of a given type(s) from a template database
    
#    Input: DATEBASE   list of templates
#           TYPES      which types should be selected
    
#    Output: array of templates of given type(s)
       
    database_trunc = list([])
    
    for type in TYPES:
        database_trunc += list([x for x in DATABASE if type in x])
    
    return np.array(database_trunc)


# ### Get Sigma

# In[6]:


def error_spectra(spec_object): 

    flux = spec_object[:,1]
    lam  = spec_object[:,0]

#For how many points do we make the lines
    num=10

    if len(flux)%num != 0:
        c = len(flux)%num
        flux = flux[:-c]
        lams = lam[:-c]
    
    else:
        lams = lam
        c = 0
    
    
    flux_new = flux.reshape((-1, num))
    lam_new  = lams.reshape((-1, num))
    m = []
    b = []
    sigma = []

    for n in range(len(lam_new)):
        r=[]
        error=[]
        
        a = np.polyfit(lam_new[n], flux_new[n], 1)
        m.append(a[0])
        b.append(a[1])
        y = m[n]*lam_new[n]+b[n]
          
        r = flux_new - y
        
        plt.plot(lam_new[n], flux_new[n], '.' )
        plt.plot(lam_new[n], y)
        plt.plot(lam_new[n], flux_new[n]-y, 'ko', markersize=1)
       
        plt.title('For n*10th Entry')
        plt.ylabel('Flux')
        plt.xlabel('Lamda')
    

    for i in r: 
        s = statistics.stdev(i)
        sigma.append(s)
    

# Here we make the error be the same size as the original lambda and then take the transpose

    error = list(np.repeat(sigma, num))
    l = [error[-1]] * c
    error = error + l

    error = np.asarray(error)
    
    return np.array([lam,error]).T


# # Compute the (Super) fit

# In[7]:


def wrapper_fit(DATABASE):

    """
    Compute the fit
    """
   
    # 1) 
    
    spec_gal    =  np.loadtxt(DATABASE['GALAXY'])
    spec_SN     =  np.loadtxt(DATABASE['SN'])
    spec_object =  np.loadtxt(DATABASE['OBJECT'])
    
    
    error       =  error_spectra(spec_object)
    spec_object_interp     =  interpolate.interp1d(spec_object[:,0], spec_object[:,1], bounds_error=False, fill_value=0)
    spec_object_err_interp =  interpolate.interp1d(error[:,0],       error[:,1],       bounds_error=False, fill_value=np.inf)
               

    
    number = 101
    #z = np.random.randint(0,20,number)/1000.
    z = np.linspace(0,0.2,number)
    chi2 = np.zeros(number)
    b = np.zeros(number)
    d = np.zeros(number)
    
    
    for i in range(len(z)):
   
       
        
        spec_gal_interp        =  interpolate.interp1d(spec_gal[:,0]*(1+z[i]),    spec_gal[:,1],    bounds_error=False, fill_value=0)
        spec_sn_interp         =  interpolate.interp1d(spec_SN[:,0]*(1+z[i]),     spec_SN[:,1],     bounds_error=False, fill_value=0)
       
    
        lambda_min   =   max([   spec_gal[:,0][0],  spec_SN[:,0][0],  spec_object[:,0][0]   ])
        lambda_max   =   min([   spec_gal[:,0][-1], spec_SN[:,0][-1], spec_object[:,0][-1]  ])
        
        
        lam          =   spec_object[:,0][ (spec_object[:,0] >= lambda_min) & (spec_object[:,0] <= lambda_max) ]
        
        
        sigma        =  spec_object_err_interp(lam)
        object_spec  =  spec_object_interp    (lam)
        
        if True:
            
            
            gal = spec_gal_interp(lam)
            sn  = spec_sn_interp(lam)
           
            
            c = 1 / ( np.sum(sn**2) * np.sum(gal**2) - np.sum(gal*sn)**2 )
            
            b[i] = c * (np.sum(gal**2)*np.sum(sn*object_spec) - np.sum(gal*sn)*np.sum(gal*object_spec))
            
            d[i] = c * (np.sum(sn**2)*np.sum(gal*object_spec) - np.sum(gal*sn)*np.sum(sn*object_spec))
            
            if b[i] < d[i]:
                chi2[i] = np.inf
            else:
                chi2[i] = np.sum(((object_spec - (b[i]*sn*10**(float(DATABASE['DUST'])*Alam(lam,1)) + d[i]*gal))/(sigma) )**2)   
                #Reduced chi2
                chi2[i] =  chi2[i]/(len(lam) - 4)
            
        
        
        #Second method, uses a curvefit routine to find the 
        else:
        
            def func(x, b, d):
                return b * spec_sn_interp(x)*10**(float(DATABASE['DUST']) * Alam(x,1)) + d * spec_gal_interp(x)    
    
            
            result = curve_fit(func, lam/(1+z[i]), object_spec, sigma=sigma, p0=[1,0], 
                              bounds=((0,0),(3,3)))
     
            popt   = result[0]
            pcov   = result[1]
            
            b[i] = popt[0]
            d[i] = popt[1]
            
            if b[i] < d[i]:     
                chi2[i] = np.inf
            else:
                chi2[i]   =  np.sum(((object_spec - func(lam, *popt))/sigma)**2)
            #Reduced chi2
                chi2[i]   =  chi2[i]/(len(lam) - 4)
            
          
    
    idx = np.argmin(chi2)
    
    
    output=table.Table(np.array([DATABASE['OBJECT'], DATABASE['GALAXY'], DATABASE['SN'], b[idx] , d[idx], z[idx], chi2[idx], 
                                  DATABASE['DUST'] ]), names=('OBJECT', 'GALAXY', 'SN', 'CONST_SN','CONST_GAL','CONST_Z','CHI2','DUST'), dtype=('S100', 'S100', 'S100','f','f','f','f','f'))
            
         
    output        
            
   
    return output
    


# # Read in spectral database

# In[8]:


#A data base will contain all templates binned at different intervals
# Here the user defines what kind of binning (how many A) he wants

templates_gal = glob.glob('binnings/20A/gal/*')
templates_gal = [x for x in templates_gal if 'CVS' not in x and 'README' not in x]
templates_gal = np.array(templates_gal)


templates_sn = glob.glob('binnings/20A/sne/**/*')
#templates_sn = glob.glob('newbank/**/*')
templates_sn = [x for x in templates_sn if 'CVS' not in x and 'README' not in x]
templates_sn = np.array(templates_sn)


#The beginning, end, and interval for the dust are decided by the user  

templates_dust = np.array([-2, -1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1 , 2])
#templates_dust = np.linspace(-2,2,11)


# ## Truncate templates SN, HG
# 

# In[ ]:


# The user decides how to truncate the supernova and host galaxy templates


#templates_sn_trunc = select_templates(templates_sn, ['/Ia/','/Ib/','/Ic/','/II/','Others'])
templates_sn_trunc = select_templates(templates_sn, ['/II/'])



#templates_gal_trunc = select_templates(templates_gal, ['/SB4'])
templates_gal_trunc = select_templates(templates_gal, ['/E','/S0','/Sa','/Sb','/SB1','/SB2','/SB3','/SB4','/SB5','/SB6','/Sc'])


# # Compute the cartesian product of SN templates, galaxy templates and extinction measurements

# In[ ]:


cartesian_product_all  =  cartesian_product(*[templates_gal_trunc[:1], templates_sn_trunc, templates_dust])
cartesian_product_all  =  table.Table(cartesian_product_all, names=('GALAXY', 'SN', 'DUST'))


# Here the user enters the template he wants to analize 
cartesian_product_all['OBJECT']=["/home/sam/Dropbox/superfit/rebinned/combined/avishay"]


cartesian_product_all


# In[ ]:


index_array=range(len(cartesian_product_all))
index_array


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'output=wrapper_fit(cartesian_product_all[0])\noutput')


# In[ ]:


get_ipython().run_cell_magic('capture', '', '\nif mp.cpu_count() > 1:\n     number_cpu\t= mp.cpu_count()/2\t# is equal to number of threads x number of physical cpus, e.g. 2x4\nelse:\n    number_cpu\t= mp.cpu_count()\nnumber_cpu\npool\t= mp.Pool(processes=int(number_cpu)*2)')


# In[ ]:


result = pool.map(wrapper_fit, cartesian_product_all)


# In[ ]:


result=table.vstack(result)


# In[ ]:


#%%capture
#result=[wrapper_fit(x) for x in cartesian_product_all]
#result=table.vstack(result)


# In[ ]:


result
result.sort('CHI2')
result


# # (Super) Graph

# In[ ]:


def visualise_match(DATABASE):

 
    spec_gal    =   np.loadtxt(DATABASE['GALAXY'][0])
    spec_SN     =   np.loadtxt(DATABASE['SN'][0])
    spec_object =   np.loadtxt(DATABASE['OBJECT'][0])
    #spec_object[:,1]*=10

    spec_gal[:,0] *= (1+DATABASE['CONST_Z'])
    spec_SN [:,0] *= (1+DATABASE['CONST_Z'])

    
    #Here a combined the names of HG and SN for the legend
    a = DATABASE['SN'][0]  + ' & ' + DATABASE['GALAXY'][0]
    
    
#Plot data

    lambda_min = max([ spec_gal[:,0][0],  spec_SN[:,0][0]   ])
    lambda_max = min([ spec_gal[:,0][-1], spec_SN[:,0][-1]  ])
    
    lam = spec_SN[:,0][ (spec_SN[:,0] >= lambda_min) & (spec_SN[:,0] <= lambda_max) ]
    
    spec_gal_interp  = interpolate.interp1d(spec_gal[:,0], DATABASE['CONST_GAL']*spec_gal[:,1], bounds_error=False, fill_value=np.nan)
    
    spec_sn_interp  =  interpolate.interp1d(spec_SN[:,0], 
                                           DATABASE['CONST_SN']  *spec_SN[:,1]*10**(float(DATABASE['DUST']) * Alam(spec_SN[:,0],1)),
                                           bounds_error=False, fill_value=np.nan)
    
    
    combined = spec_gal_interp(lam) + spec_sn_interp(lam)
    
  

    plt.figure(figsize=(7*np.sqrt(2), 7))
    
    ax = plt.subplot(111)
    ax.plot(spec_object[:,0], spec_object[:,1], 'm' , lw=5, label=DATABASE['OBJECT'][0])    
    #ax.plot(spec_gal[:,0], DATABASE['CONST_GAL'] * spec_gal[:,1], lw=1, label='Galaxy template')#: {}'.format(bestfit_galaxy))
    #ax.plot(spec_SN[:,0],  DATABASE['CONST_SN']  * spec_SN[:,1] ,  lw=3, label='SN template')#: {name} ({type}, {phase} days)'.format(type=bestfit_sn_type, name=bestfit_sn_name, phase=bestfit_sn_phase))
    ax.plot(lam, combined, 'k' ,lw=2, label= a)


    
    ax.legend(fontsize = 14)
    
    ax.set_xlabel('Observed wavelength (A)')
    ax.set_ylabel('Flux density (arbitary units)')
    
    ax.set_xlim(spec_object[0,0]-20, spec_object[-1,0]-20)
    ax.set_ylim(0, max(spec_object[:,1])*1.2)
    
    plt.show()


# In[ ]:


print(output.info())

b = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
#b = np.linspace(0,20,1)
#b = list(b)

for i in b:
    visualise_match(table.Table(result[i]))

#visualise_match(table.Table(result[0]))


# ##### This took:

# In[ ]:


end = time.time()
print(end - start, 'seconds')


# In[ ]:


table.Table(result[0])


# In[ ]:




