import numpy as np
import sncosmo
from astropy.table import Table, Column

def testCall():
	print "it worked"

def read_snana_datafile(datafile):
    """ Read in a SN data table that was written in the old 
    SNANA format, modify it, and return an astropy Table object 
    that can be handled by sncosmo functions.
    """
    metadata, datatable = sncosmo.read_snana_ascii(
        datafile, default_tablename='OBS')
    sntable = standardize_snana_data(datatable['OBS'])
    return sntable
    
def standardize_snana_data(sn, headfile=None):
    """ Modify a SN data table that was written in the old 
    SNANA format so that it can be handled by sncosmo
    """

    if 'ZEROPT' in sn.colnames and 'ZPT' not in sn.colnames:
        sn['ZEROPT'].name = 'ZPT'
        sn['ZEROPT_ERR'].name = 'ZPTERR'
    if 'ZPT' not in sn.colnames:
        sn['ZPT'] = 27.5 * np.ones(len(sn))
        
    if 'FLUXCAL' in sn.colnames and 'FLUX' not in sn.colnames:
        fluxdata = sn['FLUXCAL'] * 10 ** (0.4 * (sn['ZPT'] - 27.5))
        fluxerrdata = sn['FLUXCALERR'] * 10 ** (0.4 * (sn['ZPT'] - 27.5))
        fluxcolumn = Column(data=fluxdata, name='FLUX')
        sn.add_column(fluxcolumn)
        fluxerrcolumn = Column(data=fluxerrdata, name='FLUXERR')
        sn.add_column(fluxerrcolumn)
        #print(fluxdata)
        #print(fluxerrdata)
        #print(fluxcolumn)

    if 'FLT' in sn.colnames and 'FILTER' not in sn.colnames:
        filterdata = np.where(sn['FLT']=='Y','F125W',
                              np.where(sn['FLT']=='J','F125W','F160W'))
        filtercolumn = Column(data=filterdata, name='FILTER')
        sn.add_column(filtercolumn)
        sn.remove_column('FLT')
        #print(filterdata)
        #print(filtercolumn)

    if 'MAGSYS' not in sn.colnames:
        magsysdata = np.ones(len(sn), dtype='S2')
        magsysdata.fill('AB')
        magsyscol = Column(data=magsysdata, name='MAGSYS')
        sn.add_column(magsyscol)
        #print(magsysdata)
        #print(magsyscol)

    snstd = sncosmo.fitting.standardize_data(sn)
    #snstd = sn
    
    if headfile:
        sn.meta['HEADFILE'] = headfile
        sn.meta['PHOTFILE'] = headfile.replace('HEAD', 'PHOT')

    return Table(snstd, meta=sn.meta, copy=False)
