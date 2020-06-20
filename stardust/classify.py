import parallelize
import sys
# Dictionary of sncosmo CCSN model names and their corresponding SN sub-type
SubClassDict_SNANA = {    'ii':{    'snana-2007ms':'IIP',  # sdss017458 (Ic in SNANA)
                                    'snana-2004hx':'IIP',  # sdss000018 PSNID
                                    'snana-2005gi':'IIP',  # sdss003818 PSNID
                                    'snana-2006gq':'IIP',  # sdss013376
                                    'snana-2006kn':'IIP',  # sdss014450
                                    'snana-2006jl':'IIP',  # sdss014599 PSNID
                                    'snana-2006iw':'IIP',  # sdss015031
                                    'snana-2006kv':'IIP',  # sdss015320
                                    'snana-2006ns':'IIP',  # sdss015339
                                    'snana-2007iz':'IIP',  # sdss017564
                                    'snana-2007nr':'IIP',  # sdss017862
                                    'snana-2007kw':'IIP',  # sdss018109
                                    'snana-2007ky':'IIP',  # sdss018297
                                    'snana-2007lj':'IIP',  # sdss018408
                                    'snana-2007lb':'IIP',  # sdss018441
                                    'snana-2007ll':'IIP',  # sdss018457
                                    'snana-2007nw':'IIP',  # sdss018590
                                    'snana-2007ld':'IIP',  # sdss018596
                                    'snana-2007md':'IIP',  # sdss018700
                                    'snana-2007lz':'IIP',  # sdss018713
                                    'snana-2007lx':'IIP',  # sdss018734
                                    'snana-2007og':'IIP',  # sdss018793
                                    'snana-2007ny':'IIP',  # sdss018834
                                    'snana-2007nv':'IIP',  # sdss018892
                                    'snana-2007pg':'IIP',  # sdss020038
                                    'snana-2006ez':'IIn',  # sdss012842
                                    'snana-2006ix':'IIn',  # sdss013449
                                },
                          'ibc':{    'snana-2004fe':'Ic',
                                     'snana-2004gq':'Ic',
                                     'snana-sdss004012':'Ic',  # no IAU ID
                                     'snana-2006fo':'Ic',      # sdss013195 PSNID
                                     'snana-sdss014475':'Ic',  # no IAU ID
                                     'snana-2006lc':'Ic',      # sdss015475
                                     'snana-04d1la':'Ic',
                                     'snana-04d4jv':'Ic',
                                     'snana-2004gv':'Ib',
                                     'snana-2006ep':'Ib',
                                     'snana-2007y':'Ib',
                                     'snana-2004ib':'Ib',   # sdss000020
                                     'snana-2005hm':'Ib',   # sdss002744 PSNID
                                     'snana-2006jo':'Ib',   # sdss014492 PSNID
                                     'snana-2007nc':'Ib',   # sdss019323
                                 },
                          'ia': {'salt2-extended':'Ia'},
                      }


SubClassDict_PSNID = {
           'ii':{ 's11-2004hx':'II','s11-2005lc':'IIP','s11-2005gi':'IIP','s11-2006jl':'IIP' },
           'ibc':{ 's11-2005hl':'Ib','s11-2005hm':'Ib','s11-2006fo':'Ic', 's11-2006jo':'Ib'},
           'ia': {'salt2-extended':'Ia'},
}

from sncosmo import *
import time
import numpy as np
from astropy.io import ascii

testsnIadat = """
# time band     flux         fluxerr       zp  zpsys
 0.0 f127m 0.491947902265  0.017418231547 24.6412    ab
 0.0 f139m 0.513425670819 0.0168000764011 24.4793    ab
 0.0 f153m 0.486808758939 0.0167684488219 24.4635    ab
 0.0 f125w  2.14010106322 0.0649063974142   26.25    ab
 0.0 f140w  2.78151131439 0.0722039093523   26.46    ab
 0.0 f160w   1.6716457987 0.0594698101517   25.96    ab
"""

testsnCCdat = """
#time  band      flux         fluxerr       zp  zpsys
 0.0 f127m 0.9359 0.9674 26.47    ab
 0.0 f139m 0.8960 0.9466 26.49    ab
 0.0 f153m  1.004  1.002  26.7    ab
 0.0 f125w  3.937  1.984 28.02    ab
 0.0 f140w  5.606  2.367 28.48    ab
 0.0 f160w  3.978  1.994 28.19    ab
"""

testsnIa = ascii.read( testsnIadat )
testsnCC = ascii.read( testsnCCdat )


def mcsample( p, Ndraws, x0=None, mcsigma=0.05,
              Nburnin=100,  debug=False, *args, **kwargs ) :
    """ Crude metropolis-hastings monte carlo sampling funcion.

    The first argument is a callable function that defines
    the posterior probability at position x:  p(x).

    Positional arguments and optional keyword arguments for the function p
    may be provided at the end.  The function p will be called as
     p(x, *args, **kwargs).

    We construct a Markov Chain with  Ndraws  steps using the
    Metropolis-Hastings algorithm with a gaussian proposal distribution
    of stddev sigma.
    """
    from numpy import random
    if debug: import pdb; pdb.set_trace()

    # if user doesn't provide a starting point,
    # then draw an initial random position between 0 and 1
    if not x0 : x0 = random.uniform()
    xsamples = []
    istep = 0
    p0 = p(x0, *args, **kwargs)
    while len(xsamples) < Ndraws :
        # draw a new position from a Gaussian proposal dist'n
        x1 = random.normal( x0, mcsigma )
        p1 = p( x1, *args, **kwargs )
        # compare new against old position
        if p1>=p0 :
            # new position has higher probability, so
            # accept it unconditionally
            if istep>Nburnin : xsamples.append( x1 )
            p0=p1
            x0=x1
        else :
            # new position has lower probability, so
            # pick new or old based on relative probs.
            y = random.uniform( )
            if y<p1/p0 :
                if istep>Nburnin : xsamples.append( x1 )
                p0=p1
                x0=x1
            else :
                if istep>Nburnin : xsamples.append( x0 )
        istep +=1
    return( xsamples )



def pAv( Av, sigma=0, tau=0, R0=0, noNegativeAv=True ):
    """  Dust models:   P(Av)
    :param Av:
    :param sigma:
    :param tau:
    :param R0:
    :param noNegativeAv:
    :return:
    """
    if not np.iterable( Av ) : Av = np.array( [Av] )

    # gaussian core
    core = lambda sigma,av : np.exp( -av**2 / (2*sigma**2) )
    # Exponential tail
    tail = lambda tau,av : np.exp( -av/tau )

    if tau!=0 and noNegativeAv:
        tailOut = np.where( Av>=0, tail(tau,Av), 0 )
    elif tau!=0 :
        tailOut = tail(tau,Av)
    else :
        tailOut = np.zeros( len( Av ) )

    if sigma!=0 and noNegativeAv:
        coreOut = np.where( Av>=0, core(sigma,Av), 0 )
    elif sigma!=0 :
        coreOut = core(sigma,Av)
    else :
        coreOut = np.zeros( len( Av ) )

    if len(Av) == 1 :
        coreOut = coreOut[0]
        tailOut = tailOut[0]
    if sigma==0 : return( tailOut )
    elif tau==0 : return( coreOut )
    else : return( R0 * coreOut + tailOut )

def gauss( x, mu, sigma, range=None):
    """ Return values from a (bifurcated) gaussian.
    If sigma is a scalar, then this function returns a  symmetric
    normal distribution.

    If sigma is a 2-element iterable, then we define a bifurcated
    gaussian (i.e. two gaussians with different widths that meet with a
    common y value at x=mu)
    In this case, sigma must contain a positive value
    giving sigma for the right half gaussian, and a negative value giving
    sigma for the left half gaussian.

    If range is specified, then we include a normalization factor to
    ensure that the function integrates to unity over the given interval.
    """
    from scipy.special import erf

    if np.iterable( sigma ) :
        assert np.sign(sigma[0])!=np.sign(sigma[1]), \
            "sigma must be [+sigmaR,-sigmaL] or [-sigmaL,+sigmaR] :  " \
            "i.e. components must have opposite signs"
        sigmaL = - np.min( sigma )
        sigmaR = np.max( sigma )
    else :
        sigmaL = sigmaR = sigma

    if range is not None :
        normfactor = 2. / ( np.abs( erf( (range[0]-mu)/(np.sqrt(2)*sigmaL) ) ) + \
                            np.abs( erf( (range[1]-mu)/(np.sqrt(2)*sigmaR) ) ) )
    else :
        normfactor = 1.

    if np.iterable(x) and type(x) != np.ndarray :
        x = np.asarray( x )

    normaldist = lambda x,mu,sig : np.exp(-(x-mu)**2/(2*sig**2))/(np.sqrt(2*np.pi)*sig)
    gaussL = normfactor * 2*sigmaL/(sigmaL+sigmaR) * normaldist( x, mu, sigmaL )
    gaussR = normfactor * 2*sigmaR/(sigmaL+sigmaR) * normaldist( x, mu, sigmaR )

    if not np.iterable( x ) :
        if x <= mu :
            return( gaussL )
        else :
            return( gaussR )
    else :
        return( np.where( x<=mu, gaussL, gaussR ) )



def get_evidence(sn=testsnIa, modelsource='salt2',
                 zhost=None, zhosterr=None, t0_range=None,
                 zminmax=[0.1,2.8],
                 npoints=100, maxiter=1000, verbose=True):
    """  compute the Bayesian evidence (and likelihood distributions)
    for the given SN class using the sncosmo nested sampling algorithm.
    :return:
    """
    import os

    from scipy import interpolate, integrate
    from sncosmo import _deprecated, fitting, Model, CCM89Dust
    import time
    tstart = time.time()
    # standardize the data column names and normalize to zpt=25 AB
    #sn = _deprecated.standardize_data( sn )
    #sn = _deprecated.normalize_data( sn )

    # Define parameter bounds and priors for z, x1, c, Rv, etc
    if zhost is None :
        zhost = None
    elif isinstance(zhost,str) :
        # read in the z prior from a file giving z and p(z)
        assert os.path.isfile( zhost ), "If zprior is a string, it must be a filename"
        z,pdf = np.loadtxt( zhost, unpack=True )
        # normalize so that it integrates to unity over the allowed z range
        izgood = np.where( (zminmax[0]<z) & (z<zminmax[1]) )[0]
        pdfint = integrate.simps( pdf[izgood], z[izgood] )
        pdf = pdf / pdfint
        zprior = interpolate.interp1d( z, pdf, bounds_error=False, fill_value=0)
    else :
        if zhosterr is None :
            zhosterr = 0.1
        if np.iterable( zhosterr ) :
            assert np.sign(zhosterr[0])!=np.sign(zhosterr[1]), \
                "zphoterr must be [+err,-err] or [-err,+err] :  " \
                "i.e. components must have opposite signs"
            zhostminus = - np.min( zhosterr )
            zhostplus = np.max( zhosterr )
        else :
            zhostminus = zhostplus = zhosterr
        zmin, zmax = zminmax
        zminmax = [ max( zmin, zhost-zhostminus*5), min(zmax,zhost+zhostplus*5) ]
        def zprior( z ) :
            return( gauss( z, zhost, [-zhostminus,zhostplus], range=zminmax ) )

    if t0_range is None :
        t0_range = [sn['time'].min()-20,sn['time'].max()+20]

    if zhosterr>0.01 :
        bounds={'z':(zminmax[0],zminmax[1]),'t0':(t0_range[0],t0_range[1]) }
    else :
        bounds={'t0':(t0_range[0],t0_range[1]) }
    if modelsource.lower().startswith('salt2') :
        # define the Ia SALT2 model parameter bounds and priors
        model = Model( source=modelsource)
        if zhosterr>0.01 :
            vparam_names = ['z','t0','x0','x1','c']
        else :
            vparam_names = ['t0','x0','x1','c']
        bounds['x1'] = (-5.,5.)
        # bounds['c'] = (-0.5,3.0)
        bounds['c'] = (-0.5,5.0)  # fat red tail
        def x1prior( x1 ) :
            return( gauss( x1, 0, [-1.5,0.9], range=bounds['x1'] ) )
        def cprior( c ) :
            # return( gauss( c, 0, [-0.08,0.14], range=bounds['c'] ) )
            return( gauss( c, 0, [-0.08,0.54], range=bounds['c'] ) ) # fat red tail
        if zhost :
            priorfn = {'z':zprior, 'x1':x1prior, 'c':cprior}
        else :
            priorfn = { 'x1':x1prior, 'c':cprior }

    else :
        # define a host-galaxy dust model
        dust = CCM89Dust( )
        # Define the CC model, parameter bounds and priors
        model = Model( source=modelsource, effects=[dust],
                               effect_names=['host'], effect_frames=['rest'])

        if zhosterr>0.01 :
            vparam_names = ['z','t0','amplitude','hostebv','hostr_v']
        else :
            vparam_names = ['t0','amplitude','hostebv','hostr_v']
        # bounds['hostebv'] = (0.0,1.0)
        bounds['hostebv'] = (0.0,3.0) # fat red tail
        bounds['hostr_v'] = (2.0,4.0)
        def rvprior( rv ) :
            return( gauss( rv, 3.1, 0.3, range=bounds['host_rv'] ) )
        # TODO : include a proper Av or E(B-V) prior for CC models
        if zhost and zhosterr>0.01:
            priorfn = {'z':zprior, 'rv':rvprior }
        else :
            priorfn = { 'rv':rvprior }

    model.set(z=np.mean(zminmax))
    res, fit = fitting.nest_lc(sn, model, vparam_names, bounds,
                               guess_amplitude_bound=True,
                               priors=priorfn, minsnr=4,
                               npoints=npoints, maxiter=maxiter,
                               verbose=verbose)
    #print ("fit2: ", time.time())
    tend = time.time()
    if verbose : print("  Total Fitting time = %.1f sec"%(tend-tstart))
    return( sn, res, fit, priorfn )

def get_marginal_pdfs( res, nbins=51, verbose=True ):
    """ Given the results <res> from a nested sampling chain, determine the
    marginalized posterior probability density functions for each of the
    parameters in the model.

    :param res:  the results of a nestlc run
    :param nbins: number of bins (steps along the x axis) for sampling
       each parameter's marginalized posterior probability
    :return: a dict with an entry for each parameter, giving a 2-tuple containing
       NDarrays of length nbins.  The first array in each pair gives the parameter
       value that defines the left edge of each bin along the parameter axis.
       The second array gives the posterior probability density integrated
       across that bin.
    """
    vparam_names = res.vparam_names
    weights = res.weights
    samples = res.samples
    pdfdict = {}

    for param in vparam_names :
        ipar = vparam_names.index( param )
        paramvals = samples[:,ipar]

        if nbins>1:
            if param in res.bounds :
                parvalmin, parvalmax = res.bounds[param]
            else :
                parvalmin, parvalmax = 0.99*paramvals.min(), 1.01*paramvals.max()
            parambins = np.linspace( parvalmin, parvalmax, nbins, endpoint=True )
            binindices = np.digitize( paramvals, parambins )

            # we estimate the marginalized pdf by summing the weights of all points in the bin,
            # where the weight of each point is the prior volume at that point times the
            # likelihood, divided by the total evidence
            pdf = np.array( [ weights[np.where( binindices==ibin )].sum() for ibin in range(len(parambins)) ] )
        else :
            parambins = None
            pdf = None

        mean = (weights  * samples[:,ipar]).sum()
        std = np.sqrt( (weights * (samples[:,ipar]-mean)**2 ).sum() )

        pdfdict[param] = (parambins,pdf,mean,std)

        if verbose :
            if np.abs(std)>=0.1:
                print( '  <%s> =  %.2f +- %.2f'%( param, np.round(mean,2), np.round(std,2))  )
            elif np.abs(std)>=0.01:
                print( '  <%s> =  %.3f +- %.3f'%( param, np.round(mean,3), np.round(std,3)) )
            elif np.abs(std)>=0.001:
                print( '  <%s> =  %.4f +- %.4f'%( param, np.round(mean,4), np.round(std,4)) )
            else :
                print( '  <%s> = %.3e +- %.3e'%( param, mean, std) )


        if param == 'x0' :
            salt2 = Model( source='salt2')
            salt2.source.set_peakmag( 0., 'bessellb', 'ab' )
            x0_AB0 = salt2.get('x0')
            mBmean = -2.5*np.log10(  mean / x0_AB0 )
            mBstd = 2.5*np.log10( np.e ) *  std / mean
            mBbins = -2.5*np.log10(  parambins / x0_AB0 )

            pdfdict['mB'] = ( mBbins, pdf, mBmean, mBstd )
            print( '  <%s> =  %.3f +- %.3f'%( 'mB', np.round(mBmean,3), np.round(mBstd,3)) )

    return( pdfdict )


def inflateUncert(sn):
    #Blowing up uncertainties...
    '''
    import matplotlib.pyplot as plt
    plotting.plot_lc(sn)
    plt.show()
    '''

    signifFluxEnd = 0
    firstSigFound = False
    lastSigFound = False

    #first we iterate through backwards looking for the last significant flux measurement
    sn.reverse()
    for i in range(len(sn)):
        aRowI = sn[i]
        if( aRowI['flux'] / aRowI['fluxerr'] > 3):
            signifFluxEnd = len(sn) - i
            break
    sn.reverse()
    signifFluxEnd = sn[signifFluxEnd-1]

    #now we iterate through until first signif flux is reached
    for i in sn:
        if( i['flux'] / i['fluxerr'] > 3):
            firstSigFound = True
        if(firstSigFound):
            #once first signif flux found, start blowing up uncertainties until last signif flux
            if(i == signifFluxEnd):
                break
            else:
                i['fluxerr'] = i['fluxerr']*3
                continue
    #plotting.plot_lc(sn)
    #plt.show()
    return(sn)


def plot_marginal_pdfs( res, nbins=101, **kwargs):
    """ plot the results of a classification run
    :return:
    """
    from matplotlib import pyplot as pl

    nparam = len(res.vparam_names)
    # nrow = np.sqrt( nparam )
    # ncol = nparam / nrow + 1
    nrow, ncol = 1, nparam

    pdfdict = get_marginal_pdfs( res, nbins )

    fig = pl.gcf()
    for parname in res.vparam_names :
        iax = res.vparam_names.index( parname )+1
        ax = fig.add_subplot( nrow, ncol, iax )

        parval, pdf, mean, std = pdfdict[parname]
        ax.plot(  parval, pdf, **kwargs )
        if np.abs(std)>=0.1:
            ax.text( 0.95, 0.95, '%s  %.1f +- %.1f'%( parname, np.round(mean,1), np.round(std,1)),
                     ha='right',va='top',transform=ax.transAxes )
        elif np.abs(std)>=0.01:
            ax.text( 0.95, 0.95, '%s  %.2f +- %.2f'%( parname, np.round(mean,2), np.round(std,2)),
                     ha='right',va='top',transform=ax.transAxes )
        elif np.abs(std)>=0.001:
            ax.text( 0.95, 0.95, '%s  %.3f +- %.3f'%( parname, np.round(mean,3), np.round(std,3)),
                     ha='right',va='top',transform=ax.transAxes )
        else :
            ax.text( 0.95, 0.95, '%s  %.3e +- %.3e'%( parname, mean, std),
                     ha='right',va='top',transform=ax.transAxes )

    pl.draw()

def _parallel(args):
    modelsource,verbose,sn,zhost,zhosterr,t0_range,zminmax,npoints,maxiter,nsteps_pdf,excludetemplates=args
    
    sn, res, fit, priorfn = get_evidence(
        sn, modelsource=modelsource, zhost=zhost, zhosterr=zhosterr,
        t0_range=t0_range, zminmax=zminmax,
        npoints=npoints, maxiter=maxiter, verbose=max(0, verbose - 1))
    if nsteps_pdf:
        pdf = get_marginal_pdfs(res, nbins=nsteps_pdf,
                                verbose=max(0, verbose - 1))
    else:
        pdf = None

    #del fit._source
    outdict = {'key':modelsource,'sn': sn, 'res': res, 'fit': fit,'pdf': pdf, 'priorfn': priorfn}
    #({'sn': sn, 'res': res, 'fit': fit,'pdf': pdf, 'priorfn': priorfn})
    return(parallelize.parReturn(outdict))

def getSimTemp(theCID):
    theTempFile = 'classTest/simulatedChallange/UNBLIND_NON1A_TEMPLATE/snfit+HOST.fitres'
    theKeyFile = 'classTest/simulatedChallange/UNBLIND_NON1A_TEMPLATE/NON1A.LIST'
    CIDFound = False
    with open(theTempFile) as f:
	    content = f.readlines()
    for aLine in content:
        if str(theCID) in aLine:
            CIDFound = True
            theTempNum = aLine.split()[27]

    if(CIDFound == False):
        print ("CID not found for template exclusion, this classification should be skipped.")
    else:
        with open(theKeyFile) as f:
            content = f.readlines()
            for aLine in content[5:]:
                if theTempNum in aLine.split()[1]:
                    return aLine.split()[3]
    print("Template not found, this classification should be skipped")
    return 0

def classify(sn, zhost=1.491, zhosterr=0.003, t0_range=None,
             zminmax=[1.488,1.493], npoints=100, maxiter=10000,
             templateset='SNANA', excludetemplates=[],
             nsteps_pdf=101, verbose=True):
    """  Collect the bayesian evidence for all SN sub-classes.
    :param sn:
    :param zhost:
    :param zhosterr:
    :param t0_range:
    :param zminmax:
    :param npoints:
    :param maxiter:
    :param verbose:
    :return:
    """    
    tstart = time.time()
    if templateset.lower() == 'psnid':
        SubClassDict = SubClassDict_PSNID
    elif templateset.lower() == 'snana':
        SubClassDict = SubClassDict_SNANA
    
    iimodelnames = list(SubClassDict['ii'].keys())
    ibcmodelnames = list(SubClassDict['ibc'].keys())
    iamodelnames = list(SubClassDict['ia'].keys())

    outdict = {}
    modelProbs = SubClassDict.copy()
    allmodelnames = np.append(np.append(iamodelnames, ibcmodelnames),
                              iimodelnames)
    if excludetemplates:
        #Removing templates for simulated data (Pass in CID of SN to exclude template)
        theCID = excludetemplates.pop()
        excludetemplates.append(getSimTemp(theCID))
        if(excludetemplates[0] != 0):
            for exmod in excludetemplates:
                if exmod in allmodelnames:
                    allmodelnamelist = allmodelnames.tolist()
                    allmodelnamelist.remove(exmod)
                    allmodelnames = np.array(allmodelnamelist)

    logpriordict = {
        'ia': np.log(0.24/len(iamodelnames)),
        'ibc': np.log(0.19/len(ibcmodelnames)),
        'ii': np.log(0.57/len(iimodelnames)),
        }
    logz = {'Ia': [], 'II': [], 'Ibc': []}
    bestlogz = -np.inf

    sn = inflateUncert(sn)

#-------------------------------------------------------------------------------
    '''
    #serial code
    for modelsource in allmodelnames:
        if verbose >1:
            dt = time.time() - tstart
            print('------------------------------')
            print("model: %s  dt=%i sec" % (modelsource, dt))
        sn, res, fit, priorfn = get_evidence(
            sn, modelsource=modelsource, zhost=zhost, zhosterr=zhosterr,
            t0_range=t0_range, zminmax=zminmax,
            npoints=npoints, maxiter=maxiter, verbose=max(0, verbose - 1))

        if nsteps_pdf:
            pdf = get_marginal_pdfs(res, nbins=nsteps_pdf,
                                    verbose=max(0, verbose - 1))
        else:
            pdf = None
        outdict[modelsource] = {'sn': sn, 'res': res, 'fit': fit,
                                'pdf': pdf, 'priorfn': priorfn}
        if res.logz>bestlogz :
            outdict['bestmodel'] = modelsource
            bestlogz = res.logz

        # multiply the model evidence by the sub-type prior
        if modelsource in iimodelnames:
            logprior = logpriordict['ii']
            logz['II'].append(logprior + res.logz )
            modelProbs['ii'][modelsource] = res.logz
        elif modelsource in ibcmodelnames:
            logprior = logpriordict['ibc']
            logz['Ibc'].append(logprior + res.logz)
            modelProbs['ibc'][modelsource] = res.logz
        elif modelsource in iamodelnames:
            logprior = logpriordict['ia']
            logz['Ia'].append(logprior + res.logz)
            modelProbs['ia'][modelsource] = res.logz
        
    if(verbose):
        import pprint
        print(pprint.pprint(modelProbs))

    # sum up the evidence from all models for each sn type
    logztype = {}
    for modelsource in ['II', 'Ibc', 'Ia']:
        logztype[modelsource] = logz[modelsource][0]
        for i in range(1, len(logz[modelsource])):
            logztype[modelsource] = np.logaddexp(
                logztype[modelsource], logz[modelsource][i])
    '''
#-------------------------------------------------------------------------------
    #parallelized code
    
    res=parallelize.foreach(allmodelnames,_parallel,[verbose,sn,zhost,zhosterr,t0_range,zminmax,npoints,maxiter,nsteps_pdf,excludetemplates])
    print('------------------------------')
    dt = time.time() - tstart
    print("dt=%i sec" %dt)
    
    for modelsource in allmodelnames:
        outdict[modelsource] = {'sn': res[modelsource]['sn'], 'res': res[modelsource]['res'],
                                'pdf': res[modelsource]['pdf']}
        if res[modelsource]['res']['logz']>bestlogz :
            outdict['bestmodel'] = modelsource
            bestlogz = res[modelsource]['res']['logz']

        # multiply the model evidence by the sub-type prior
        if modelsource in iimodelnames:
            logprior = logpriordict['ii']
            logz['II'].append(logprior + res[modelsource]['res']['logz'] )
            modelProbs['ii'][modelsource] = res[modelsource]['res']['logz']
        elif modelsource in ibcmodelnames:
            logprior = logpriordict['ibc']
            logz['Ibc'].append(logprior + res[modelsource]['res']['logz'])
            modelProbs['ibc'][modelsource] = res[modelsource]['res']['logz']
        elif modelsource in iamodelnames:
            logprior = logpriordict['ia']
            logz['Ia'].append(logprior + res[modelsource]['res']['logz'])
            modelProbs['ia'][modelsource] = res[modelsource]['res']['logz']
        
    if(verbose):
        import pprint
        print(pprint.pprint(modelProbs))

    # sum up the evidence from all models for each sn type
    logztype = {}
    for modelsource in ['II', 'Ibc', 'Ia']:
        logztype[modelsource] = logz[modelsource][0]
        for i in range(1, len(logz[modelsource])):
            logztype[modelsource] = np.logaddexp(
                logztype[modelsource], logz[modelsource][i])
    
#-------------------------------------------------------------------------------
    # define the total evidence (final denominator in Bayes theorem) and then
    # the classification probabilities
    logzall = np.logaddexp(np.logaddexp(
        logztype['Ia'], logztype['Ibc']), logztype['II'])
    pIa = np.exp(logztype['Ia'] - logzall)
    pIbc = np.exp(logztype['Ibc'] - logzall)
    pII = np.exp(logztype['II'] - logzall)
    outdict['pIa'] = pIa
    outdict['pIbc'] = pIbc
    outdict['pII'] = pII
    outdict['logztype'] = logztype
    outdict['logzall'] = logzall

    if(verbose):
        print("pIa: "),
        print(pIa)
        print("pIbc: "),
        print(pIbc)
        print("pII: "),
        print(pII)
    return outdict

def plot_maxlike_fit( fitdict, **kwarg ):
    sn = fitdict['sn']
    fit = fitdict['fit']
    res = fitdict['res']
    paramnames = res.vparam_names
    errors = res.errors
    errdict = dict([ [paramnames[i],errors[i]] for i in range(len(errors))] )
    plot_lc( sn, model=fit, errors=errdict, **kwarg )


def plot_fits(classdict, nshow=2, verbose=False, **kwarg ):
    from matplotlib import cm

    plotting._cmap_wavelims = [5000, 17500]
    plotting._cmap = cm.gist_rainbow

    bestIamod, bestIbcmod, bestIImod = get_bestfit_modelnames(
        classdict, verbose=verbose)
    fitIa = classdict[bestIamod]['fit']
    fitIbc = classdict[bestIbcmod]['fit']
    fitII = classdict[bestIImod]['fit']

    sn = classdict[bestIamod]['sn']
    if nshow == 3:
        plot_lc( sn, model=[fitIa,fitIbc,fitII], model_label=['Ia','Ib/c','II'], **kwarg )
    elif nshow == 2:
        plot_lc( sn, model=[fitIa,fitIbc], model_label=['Ia','Ib/c'], **kwarg )
    elif nshow == 1:
        plot_lc(sn, model=[fitIa], model_label=['Ia'], **kwarg)


def get_bestfit_modelnames(classdict, verbose=True):
    """ Extract the name of the best-fit model for each sub-class (Ia,Ib/c,II)
    by comparing the log(Z) likelihoods in the classification results.

    :param classdict: a dictionary of classification results
    :return:
    """
    IImodlist = [modname for modname in classdict.keys() if modname in
                 SubClassDict_SNANA['ii'].keys()]
    IIlogzlist = [classdict[modname]['res']['logz'] for modname in IImodlist]
    ibestII = np.argmax(IIlogzlist)
    bestIImod = IImodlist[ibestII]
    if verbose:
        print('Best II model : %s' % bestIImod)

    Ibcmodlist = [modname for modname in classdict.keys() if modname in
                  SubClassDict_SNANA['ibc'].keys()]
    Ibclogzlist = [classdict[modname]['res']['logz'] for modname in Ibcmodlist]
    ibestIbc = np.argmax(Ibclogzlist)
    bestIbcmod = Ibcmodlist[ibestIbc]
    if verbose:
        print('Best Ib/c model : %s' % bestIbcmod)

    if 'salt2-extended' in classdict.keys() :
        bestIamod = 'salt2-extended'
    else:
        bestIamod = 'salt2'
    return bestIamod, bestIbcmod, bestIImod


def plot_color_vs_redshift(modelname, bandpass1, bandpass2, t=0,
                           zrange=[0.01,2.5], zpsys='AB',
                           parameters=None, **plotkwargs):
    """ For the given sncosmo model, plot color at time t (relative to
    the model t0... typically peak brightness) as a function of
    redshift over the given zrange.
    """

def testClassification():
    import imp
    read_des_datfile = imp.load_source("read_des_datfile","classTest/read_des_datfile.py")

    theNpoints = 20
    theMaxIter = 5000
    theNsteps = 1000
    theTemplate = 'psnid'

    testSNfile = "classTest/simulatedChallange/DES_SN180720.DAT"
    zerror = getTheZerr(testSNfile) #this is because the snana read strips the zerror
    metadata, data = read_des_datfile.read_des_datfile(testSNfile)
    theID = metadata["SNID"]

    print("Classifying, this may take awhile...")
    test_out = classify(data, zhost=metadata['HOST_GALAXY_PHOTO-Z'], zhosterr=zerror, 
                        zminmax=[metadata['HOST_GALAXY_PHOTO-Z'] - (2*zerror),
                        metadata['HOST_GALAXY_PHOTO-Z'] + (2*zerror)], npoints=theNpoints, maxiter=theMaxIter, 
                        nsteps_pdf=theNsteps, templateset=theTemplate, excludetemplates=[theID], verbose=0)

    print("Successful classification!")

def getTheZerr(theFile): #gets the z error for the host galaxy
	with open(theFile) as f:
		content = f.readlines()
	amatch = [s for s in content if "HOST_GALAXY_PHOTO-Z" in s]
	thematch = amatch[0].split("+-")
	thereturn = float(thematch[1].strip())
	return thereturn

testClassification()