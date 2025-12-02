#%%
# %%
import finesse
import numpy as np
from finesse.analysis.actions import FrequencyResponse3, FrequencyResponse4
import matplotlib.pyplot as plt
import finesse.components as fc
import finesse.detectors as fd
from finesse.utilities.maps import circular_aperture

import finesse.analysis.actions as fa
from finesse.knm import Map
import finesse.thermal.hello_vinet as hv
import yaml
from finesse.materials import FusedSilica

finesse.init_plotting()

import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import sys
sys.path.append('/Users/muusse/HWS/')
from HS_Centroids import HS_Centroids
from HS_Gradients import HS_Gradients
import HSM_WFN_dev as HSM_WFN
from scipy.optimize import curve_fit 
from scipy.interpolate import RegularGridInterpolator
import os
from datetime import datetime


#  codes 
def bbbq(hsg,rat = 4):
    """_summary_

    Args:
        hsg (arrary): HS_Gradients class
        rat (int, optional): ratio of the max quiver to mean quiver where quiver gets removed. Defaults to 4.

    Returns:
        _type_:HS_Gradients class object with large quivers removed
    """
    abs_mag = np.sqrt(hsg.gradients[:,0]**2 + hsg.gradients[:,1]**2)
    mean_len = np.mean(abs_mag)
    max_len = np.max(abs_mag)
    ratio = max_len/mean_len

    if ratio > rat:
        location = np.where( abs_mag > rat*np.std(abs_mag))
        hsg.gradients = np.transpose([np.delete(hsg.gradients[:,ii],location) for ii in range(np.shape(hsg.gradients)[1])])
    return hsg

def rp(hsg):
    """removes prism from gradients

    Args:
        hsg (arrary): HS_Gradients class


    Returns:
        class :HS_Gradients class object with prism removed

    """
    hsg.gradients[:,0]-=hsg.gradients[:,0].mean()
    hsg.gradients[:,1]-=hsg.gradients[:,1].mean()
    return hsg


def twodG(xv,x0,y0,a,b,c,d):

    """Create 2D gaussian for fitting

    Args:
        xv (array): (2,n) array of x,y grid positions
        x0 (float): centre offset in x
        y0 (float): centre offset in x
        a (float): magnitude factor
        b (float): beam radius (wx) squared
        c (float): beam radius (wy) squared
        d (float): z offset

    Returns:
        float: 2D gaussian
    """
    x,y = xv[0],xv[1]
    return a*np.exp((x-x0)**2/b+(y-y0)**2/c) +d

def find_centerpoint(wfn):
    """find center point of wavefront deformation a 2d curve by fitting to a 2D gaussian 

    Args:
        wfn (array): (3, nxn) wavefront coordinates and wavefron t values in form [x_coord,y_coord, WF] as given by HSM_WFN.calculate_wfn_centroidbound

    Returns: 
        array: Position of centre of wavefront deformation in units of x_coord,y_coord
    """
    p0 = curve_fit(twodG,np.vstack([wfn[0].flatten(),wfn[1].flatten()]),np.nan_to_num(wfn[2].flatten()), p0 =[0,0,1e-7,0,0,0])
    return p0[0][0],p0[0][1]

def overlap_curvature_coefficients(x, y, z, c, weight_spot: float, f=False):
    """Computes the amount of x and y curvature terms present in a map's displacement
    data.

    This is computed by evaluating a weighted Hermite polynomial overlap integral in an
    efficient manner.

    Parameters
    ----------
    xn, yn : array_like
        1D Array of x and y describing the 2D plane of `z`
    z : array_like
        2D optical path depth [metres]
    c : array_like      
        centre point of deformation, [2,1] array
    weight_spot : float
        Beam spot size to weight over
    f : True/False
        if True determine lens focal insaid of Roc

    Returns
    -------
    k20, k02 : complex
        Complex-valued overlap coefficients for the HG20 and HG02 mode
    """
    # Shifting to centre
    F = RegularGridInterpolator((x-c[0],y-c[1]), z, fill_value = 0, bounds_error =False)
    x = np.linspace(x[0] - abs(c[0]), x[-1] - abs(c[0]), len(x))
    y = np.linspace(y[0] - abs(c[1]), y[-1] - abs(c[1]), len(x))
    xx,yy=np.meshgrid(x,y)
    z = F(np.c_[xx.flatten() ,yy.flatten()]).reshape((len(x),len(x)))

    # Normalisation constant
    weight_spot /= np.sqrt(2)
    dx = (x[1] - x[0]) / weight_spot
    dy = (y[1] - y[0]) / weight_spot
    # Spot size weightings
    Wx = np.exp(-((x / weight_spot) ** 2))
    Wy = np.exp(-((y / weight_spot) ** 2))
    # 2nd order HG mode
    Hn = ((2 * x / weight_spot) ** 2 - 2) * Wx
    Hm = ((2 * y / weight_spot) ** 2 - 2) * Wy
    # Compute the overlap integrals
    k20 = np.dot((z @ Hn), Wy) * dx / (np.sqrt(np.pi) * 2**3) * dy / (np.sqrt(np.pi))
    k02 = np.dot((z @ Wx), Hm) * dx / (np.sqrt(np.pi) * 2**3) * dy / (np.sqrt(np.pi))
    if f==True:
        return (
        # convert back to normal x units in displacement map and then to Rc
        (4 * k20 / weight_spot / weight_spot * 4)**-1,
        (4 * k02 / weight_spot / weight_spot * 4)**-1, 
    )
    else:   
        return (
            # convert back to normal x units in displacement map and then to Rc
            (4 * k20 / weight_spot / weight_spot * 2)**-1,
            (4 * k02 / weight_spot / weight_spot * 2)**-1, 
        )

def get_rc(wfn, weight_spot=3e-3,f=False):
    """ using WF get Radius of curvature

    Args:
    wfn : (array)
        (3, nxn) wavefront coordinates and wavefron t values in form [x_coord,y_coord, WF] as given by HSM_WFN.calculate_wfn_centroidbound
    weight_spot : float
        Beam spot size to weight over. Defaults to 3e-3.    finesse uses (model.HR_port.i.qx.w + model.HR_port.i.qy.w) / 2
    f : True/False
        if True determine lens focal insaid of RocDefaults to False.

    Returns: (array)
        radius of curvature in x and y 
    """
    x0,y0 =find_centerpoint(wfn)

    rx,ry = overlap_curvature_coefficients(wfn[0,0],wfn[1,:,0],np.nan_to_num(wfn[2]).T,c = [x0,y0], weight_spot= weight_spot,f=f)
    return rx,ry



# %%TEST
a = 0.095/2 # mirror radius
h = 60e-3  # mirror thickness
w = 14.6e-3 # spot size radius
r = np.linspace(-a, a, 50) # radial points
z = np.linspace(-h/2, h/2, 100) # longitudinal points
material = FusedSilica

U_s_coat = hv.surface_deformation_coating_heating_HG00(r, a, h, w, FusedSilica)
opl_U_s_coat = U_s_coat*(FusedSilica.nr-1)
#%% make map
P_ab = 1
m_r=a
rr  = np.linspace(-m_r,m_r,1000)
rx,ry = np.meshgrid(rr,rr)
oplxy = np.interp(np.sqrt(rx.flatten()**2+ry.flatten()**2),r,opl_U_s_coat).reshape((len(rr),len(rr)))

test_map =  Map(rr,rr,opd = oplxy*P_ab, amplitude=circular_aperture(rr, rr, m_r))
print(f' finesse {test_map.get_radius_of_curvature_reflection(w)}')

rc_sophie = overlap_curvature_coefficients(rr,rr,oplxy,np.array([0,0]), w)
print(f' sophie code {rc_sophie}')


# %% NEW fucntion
get_rc(np.array([rx,ry,oplxy]),weight_spot=w)
# %%
