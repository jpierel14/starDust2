import numpy as np
import sncosmo
from astropy.table import Table, Column


def read_des_datfile(datafile):
    """ Read in a SN data table that was written in the old 
    SNANA format, modify it, and return an astropy Table object 
    that can be handled by sncosmo functions.
    """
    metadata, datatable = sncosmo.read_snana_ascii(
        datafile, default_tablename='OBS')
    sntable = standardize_snana_data(datatable['OBS'])
    return metadata, sntable


def standardize_data(data):
    """Standardize photometric data by converting to a structured numpy array
    with standard column names (if necessary) and sorting entries in order of
    increasing time.

    Parameters
    ----------
    data : `~astropy.table.Table` or `~numpy.ndarray` or `dict`

    Returns
    -------
    standardized_data : `~numpy.ndarray`
    """

    #warn_once('standardize_data', '1.5', '2.0',
    #          'This function not intended for public use; open an issue at '
    #          'https://github.com/sncosmo/sncosmo/issues if you need this '
    #          'functionality.')

    if isinstance(data, Table):
        data = np.asarray(data)

    if isinstance(data, np.ndarray):
        colnames = data.dtype.names

        # Check if the data already complies with what we want
        # (correct column names & ordered by date)
        if (set(colnames) == set(PHOTDATA_ALIASES.keys())
                and np.all(np.ediff1d(data['time']) >= 0.)):
            return data

    elif isinstance(data, dict):
        colnames = data.keys()

    else:
        raise ValueError('Unrecognized data type')

    # Create mapping from lowercased column names to originals
    lower_to_orig = dict([(colname.lower(), colname) for colname in colnames])

    # Set of lowercase column names
    lower_colnames = set(lower_to_orig.keys())

    orig_colnames_to_use = []
    for aliases in PHOTDATA_ALIASES.values():
        i = lower_colnames & aliases
        if len(i) != 1:
            raise ValueError('Data must include exactly one column from {0} '
                             '(case independent)'.format(', '.join(aliases)))
        orig_colnames_to_use.append(lower_to_orig[i.pop()])

    if isinstance(data, np.ndarray):
        new_data = data[orig_colnames_to_use].copy()
        new_data.dtype.names = list(PHOTDATA_ALIASES.keys())

    else:
        new_data = OrderedDict()
        for newkey, oldkey in zip(PHOTDATA_ALIASES.keys(),
                                  orig_colnames_to_use):
            new_data[newkey] = data[oldkey]

        new_data = dict_to_array(new_data)

    # Sort by time, if necessary.
    if not np.all(np.ediff1d(new_data['time']) >= 0.):
        new_data.sort(order=['time'])

    return new_data


def standardize_snana_data(sn, headfile=None):
    """ Modify a SN data table that was written in the old 
    SNANA format so that it can be handled by sncosmo
    """
    if 'MJD' in sn.colnames:
        sn['MJD'].name = 'time'
        timedata = sn['time']
        timecolumn = Column(data=timedata, name='time')

    if 'FLT' in sn.colnames and 'FILTER' not in sn.colnames:
        filterdata = np.where(sn['FLT'] == 'g', 'desg',
                              np.where(sn['FLT'] == 'r', 'desr',
                                       np.where(sn['FLT'] == 'i', 'desi',
                                                np.where(
                                                    sn['FLT'] == 'z', 'desz',
                                                    '?'))))
        filtercolumn = Column(data=filterdata, name='band')
        sn.add_column(filtercolumn)
        sn.remove_column('FLT')
        sn.remove_column('FIELD')
    if 'MAG' in sn.colnames:
        sn.remove_column('MAG')
        sn.remove_column('MAGERR')

    if 'ZEROPT' in sn.colnames and 'ZPT' not in sn.colnames:
        sn['ZEROPT'].name = 'zp'
    if 'ZPT' not in sn.colnames:
        sn['zp'] = 27.5 * np.ones(len(sn))

    if 'FLUXCAL' in sn.colnames and 'FLUX' not in sn.colnames:
        fluxdata = sn['FLUXCAL'] * 10**(0.4 * (sn['zp'] - 27.5))
        fluxerrdata = sn['FLUXCALERR'] * 10**(0.4 * (sn['zp'] - 27.5))
        fluxcolumn = Column(data=fluxdata, name='flux')
        sn.add_column(fluxcolumn)
        fluxerrcolumn = Column(data=fluxerrdata, name='fluxerr')
        sn.add_column(fluxerrcolumn)
        sn.remove_column('FLUXCAL')
        sn.remove_column('FLUXCALERR')
        sn.remove_column('zp')

    if 'ZEROPT' in sn.colnames and 'ZPT' not in sn.colnames:
        sn['ZEROPT'].name = 'zp'
    if 'ZPT' not in sn.colnames:
        sn['zp'] = 27.5 * np.ones(len(sn))

    if 'MAGSYS' not in sn.colnames:
        abVals = ['ab' for i in range(len(sn))]
        magsyscol = Column(data=abVals, name='zpsys')
        sn.add_column(magsyscol)

    #snstd = sncosmo._deprecated.standardize_data(sn)
    snstd = sn

    if headfile:
        sn.meta['HEADFILE'] = headfile
        sn.meta['PHOTFILE'] = headfile.replace('HEAD', 'PHOT')

    return Table(snstd, meta=sn.meta, copy=False)
