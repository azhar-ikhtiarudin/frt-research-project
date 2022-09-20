import numpy as np
import matplotlib as plt
import pandas as pd
from scipy.stats import truncnorm
from matplotlib.colors import LogNorm
from scipy.stats import multivariate_normal

class Detector:
    '''
    The detector class is used to store key detector parameters. This will be passed to the various 'event' functions.
    Note units are keV, cm, V, and seconds.
    '''
    def __init__(self, description, wval=26.4/1000, D_xy=0.04, D_z=0.04,
            mobility=1.7, Edrift=100, pitch_x=0.1, pitch_y=0.1, 
            samplerate=1e-3, gain_mean=10**5, PSFmean=0.02, PSFstd=0.01,
            gain_sigma_t=10**-8):
        '''
        Default gas values are for Argon gas
        '''
        self.description = description
        self.wval = wval
        self.D_xy = D_xy
        self.D_z = D_z
        self.mobility = mobility
        self.Ed = Edrift
        self.pitch_x = pitch_x
        self.pitch_y = pitch_y
        self.samplerate = samplerate
        self.gain_mean = gain_mean
        self.PSFmean = PSFmean
        self.PSFstd = PSFstd
        self.gain_sigma_t = gain_sigma_t
        #Mobility is the proportionality constant between the E field and the velocity
        self.vdrift = mobility*Edrift
                

def get_truncated_norm(mean=0, sd=1, low=0, upp=10, nsamp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(size=nsamp)


def generate_carriers(PrimaryEvt, det):
    '''
    Generate the charge carriers from the primary event.
    
    Inputs:

    - PrimaryEvt is a Pandas DataFrame with energy deposit values stored in a column named 'Edep'.

    - det is a TPCevt.Detector object, with the w_value set.

    The function will make a new column 'NIP' in PrimaryEvt in place, storing the number of ion pairs.
    '''
    PrimaryEvt['NIP'] = np.random.poisson(PrimaryEvt['Edep']/det.wval)
    return

def driftsigma_trans(z, det):
    return np.sqrt(2*z*det.D_xy/det.vdrift)

def driftsigma_long(z, det):
    return np.sqrt(2*z*det.D_z/det.vdrift)


def drift_carriers(PrimaryEvt, det):
    '''
    Drift the primary carriers down to the gain stage.

    Inputs:

    - PrimaryEvt is a Pandas DataFrame with at least (x,y,z,Edep,NIP).

    - det is an instance of TPCevt.Detector, with the gas properties and drift field set.

    Returns a dataframe of drifted carriers with (index in PrimaryEvt, x, y, dt)
    '''
    Ndrifted = PrimaryEvt['NIP'].sum()
    drifteddict = {'idx_PrimaryEvt': np.empty(Ndrifted, dtype=np.int), 'x': np.empty(Ndrifted, dtype=np.float),
                  'y': np.empty(Ndrifted, dtype=np.float), 'dt': np.empty(Ndrifted, dtype=np.float)}
    counter = 0
    for i in range(len(PrimaryEvt)):
        numtodrift = PrimaryEvt['NIP'][i]
        thisz = PrimaryEvt['z'][i]
        drifteddict['idx_PrimaryEvt'][counter:(counter+numtodrift)] = i
        drifteddict['x'][counter:(counter+numtodrift)] = PrimaryEvt['x'][i] + np.random.normal(loc=0,
                                                                scale=driftsigma_trans(thisz, det),
                                                                size=numtodrift)
        drifteddict['y'][counter:(counter+numtodrift)] = PrimaryEvt['y'][i] + np.random.normal(loc=0,
                                                                scale=driftsigma_trans(thisz, det),
                                                                size=numtodrift)
        drifteddict['dt'][counter:(counter+numtodrift)] = (thisz + np.random.normal(loc=0,
                                                                scale=driftsigma_long(thisz, det),
                                                                size=numtodrift))/det.vdrift
        counter += numtodrift
    
    return pd.DataFrame(drifteddict)

def gain_and_readout(DriftedEvt, det, nsigma_extend=5):
    '''
    Apply avalanche gain and read out the event. Only reads out a subset of co-ords about the track.
    
    Inputs:

    - DriftedEvt is a dataframe containing the x,y,dt co-ordinates of every drifted carrier.

    - det is an instance of TPCevt.Detector, with the gain/readout info set.

    - nsigma_extend is the # of point spread function std to extend by about the track extremities.

    Returns a tuple of (4D meshgrid of co-ordinates, Number of electrons at those co-ordinates)
    '''
    minvals = DriftedEvt.min()
    maxvals = DriftedEvt.max()
    ReadoutGrid = np.mgrid[np.floor((minvals['x']- det.PSFmean - 
                                    nsigma_extend*det.PSFstd)/det.pitch_x)*det.pitch_x:
                            np.ceil((maxvals['x'] + det.PSFmean + 
                                    nsigma_extend*det.PSFstd)/det.pitch_x)*det.pitch_x:
                            det.pitch_x,
                            np.floor((minvals['y'] - det.PSFmean - 
                                nsigma_extend*det.PSFstd)/det.pitch_y)*det.pitch_y:
                            np.ceil((maxvals['y'] + det.PSFmean + 
                                nsigma_extend*det.PSFstd)/det.pitch_y)*det.pitch_y:
                            det.pitch_y,
                            np.floor((minvals['dt'] - 
                                nsigma_extend*det.gain_sigma_t)/det.samplerate)*det.samplerate:
                            np.ceil((maxvals['dt'] + 
                                nsigma_extend*det.gain_sigma_t)/det.samplerate)*det.samplerate:
                            det.samplerate]
    pos = np.stack(ReadoutGrid, axis=3)
    ReadoutEvt = None
    for i in range(len(DriftedEvt)):
        thisGain = np.random.exponential(scale=det.gain_mean)
        rv = multivariate_normal([DriftedEvt.iloc[i]['x'], DriftedEvt.iloc[i]['y'], 
            DriftedEvt.iloc[i]['dt']], np.diag([det.PSFstd, det.PSFstd,det.gain_sigma_t]))
        if ReadoutEvt is None:
            #The factor pitch_x*pitch_y*sample_rate converts from probability density 
            #to 'normalised' probability
            ReadoutEvt = rv.pdf(pos)*det.pitch_x*det.pitch_y*det.samplerate*thisGain
        else:
            ReadoutEvt += rv.pdf(pos)*det.pitch_x*det.pitch_y*det.samplerate*thisGain

    return ReadoutGrid, ReadoutEvt

#Putting this on the backburner for now...
#def plot_readout_projection(ReadoutGrid, ReadoutEvt, xidx, yidx):
#    '''
#    Plot a quick projection of the readout event.
#
#    Inputs are the ReadoutGrid and ReadoutEvt returned by gain_and_readout, and the indices to project.
#    '''
#    fig = plt.figure()
#    namemap = {0:'x', 1:'y', 2:'dt'}
#    slicex
#    plt.contourf(ReadoutGrid[xidx][:,:,0],ReadoutGrid[1][:,:,0], np.sum(ReadoutEvt, axis=2))
