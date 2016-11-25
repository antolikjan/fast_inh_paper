#==================================#
#  OR map contour indentification  #
#==================================#
import sys
import numpy as np
import cmath, math, itertools
import pylab as plt
from numpy.fft.fftpack import fft2
from numpy.fft.helper import fftshift
from scipy.optimize import curve_fit








class WarningCounter(object):
    """
    A simple class to count 'divide by zero' and 'invalid value'
    exceptions to allow a suitable warning message to be generated.
    """
    def __init__(self):
        self.div_by_zeros=0
        self.invalid_values=0

    def __call__(self, errtype, flag):
        if errtype == "divide by zero":  self.div_by_zeros += 1
        elif errtype == "invalid value": self.invalid_values += 1

    def warn(self):
        total_events = self.div_by_zeros + self.invalid_values
        if total_events == 0: return
        info = (total_events, self.div_by_zeros, self.invalid_values)
        self.div_by_zeros=0; self.invalid_values=0
        message = ("Warning: There were %d invalid intersection events:"
                   "\n\tNumpy 'divide by zero' events: %d"
                   "\n\tNumpy 'invalid value' events: %d\n")
        sys.stderr.write( message %info)




#======================#
# Image transformation #
#======================#

def normalize_polar_channel(polar_channel):
    """This functions normalizes an OR map (polar_channel) taking into
    account the region of interest (ROI). The ROI is specified by
    values set to 99. Note that this functionality is implemented to
    reproduce the experimental approach and has not been tested (not
    required for Topographica simulations)
    """

    def grad(r):
        (r_x,r_y) = np.gradient(r)
        (r_xx,r_xy)=np.gradient(r_x);
        (r_yx,r_yy)=np.gradient(r_y);
        return r_xx**2+ r_yy**2 + 2*r_xy**2

    roi = np.ones(polar_channel.shape) # Set ROI to 0 to ignore values of -99.
    roi[roi == -99] = 0                # In Matlab: roi(find(z==-99))=0

    fst_grad = grad(roi)
    snd_grad = grad(fst_grad)

    snd_grad[snd_grad != 0] = 1  # Find non-zero elements in second grad and sets to unity
    roi[snd_grad == 1] = 0       # These elements now mask out ROI region (set to zero)

    ind = (polar_channel != 99)                    # Find the unmasked coordinates
    normalisation = np.mean(np.abs(polar_channel)) # The complex abs of unmasked
    return polar_channel / normalisation           # Only normalize with unmasked


def power_spectrum(pref, peak_val=1.0):
    """
    Computes the FFT power spectrum of the orientation preference.
    """
    fft_spectrum = abs(fftshift(fft2(pref-0.5, s=None, axes=(-2,-1))))
    fft_spectrum = 1 - fft_spectrum # Inverted spectrum by convention
    zero_min_spectrum = fft_spectrum - fft_spectrum.min()
    spectrum_range = fft_spectrum.max() - fft_spectrum.min()
    normalized_spectrum = (peak_val * zero_min_spectrum) / spectrum_range
    return normalized_spectrum


def polar_preference(pref):
   """
   Converts the preference to polar representation.  Preference values
   supplied should be in the range 0-1.
   """
   polarfn = lambda x: cmath.rect(1.0, 2*x*math.pi)
   polar_vecfn = np.vectorize(polarfn)
   return polar_vecfn(pref)

#=========================#
# Pinwheeli dentification #
#=========================#


def remove_path_duplicates(vertices):
    "Removes successive duplicates along a path of vertices."
    zero_diff_bools = np.all(np.diff(vertices, axis=0) == 0, axis=1)
    duplicate_indices, = np.nonzero(zero_diff_bools)
    return np.delete(vertices, duplicate_indices, axis=0)

def polarmap_contours(polarmap):  # Example docstring
   """
   Identifies the real and imaginary contours in a polar map.  Returns
   the real and imaginary contours as 2D vertex arrays together with
   the pairs of contours known to intersect. The coordinate system is
   normalized so x and y coordinates range from zero to one.
   Contour plotting requires origin='upper' for consistency with image
   coordinate system.
   """
   # Convert to polar and normalise
   normalized_polar = normalize_polar_channel(polarmap)
   figure_handle = plt.figure()
   # Real component
   re_contours_plot = plt.contour(normalized_polar.real, levels=[0], origin='upper')
   re_path_collections = re_contours_plot.collections[0]
   re_contour_paths = re_path_collections.get_paths()
   # Imaginary component
   im_contours_plot = plt.contour(normalized_polar.imag, levels=[0], origin='upper')
   im_path_collections = im_contours_plot.collections[0]
   im_contour_paths = im_path_collections.get_paths()
   plt.close(figure_handle)
   intersections = [ (re_ind, im_ind)
                     for (re_ind, re_path) in enumerate(re_contour_paths)
                     for (im_ind, im_path) in enumerate(im_contour_paths)
                    if im_path.intersects_path(re_path)]

   (ydim, xdim) = polarmap.shape
   # Contour vertices  0.5 pixel inset. Eg. (0,0)-(48,48)=>(0.5, 0.5)-(47.5, 47.5)
   # Returned values will not therefore reach limits of 0.0 and 1.0
   re_contours =  [remove_path_duplicates(re_path.vertices) / [ydim, xdim] \
                       for re_path in re_contour_paths]
   im_contours =  [remove_path_duplicates(im_path.vertices) / [ydim, xdim] \
                       for im_path in im_contour_paths]
   return (re_contours,  im_contours, intersections)



def find_intersections(contour1, contour2):
    """
    Vectorized code to find intersections between contours. All
    successive duplicate vertices along the input contours must be
    removed to help avoid division-by-zero errors.
    There are cases were no intersection exists (eg. parallel lines)
    where division by zero and invalid value exceptions occur. These
    exceptions should be caught as warnings: these edge cases are
    unavoidable with this algorithm and do not indicate that the
    output is erroneous.
    """
    amin = lambda x1, x2: np.where(x1<x2, x1, x2)       # Elementwise min selection
    amax = lambda x1, x2: np.where(x1>x2, x1, x2)       # Elementwise max selection
    aall = lambda abools: np.dstack(abools).all(axis=2) # dstacks, checks True depthwise
    # Uses delta (using np.diff) to find successive slopes along path
    slope = lambda line: (lambda d: d[:,1]/d[:,0])(np.diff(line, axis=0))
    # Meshgrids between both paths (x and y). One element sliced off end/beginning
    x11, x21 = np.meshgrid(contour1[:-1, 0], contour2[:-1, 0])
    x12, x22 = np.meshgrid(contour1[1:, 0], contour2[1:, 0])
    y11, y21 = np.meshgrid(contour1[:-1, 1], contour2[:-1, 1])
    y12, y22 = np.meshgrid(contour1[1:, 1], contour2[1:, 1])
    # Meshgrid of all slopes for both paths
    m1, m2 = np.meshgrid(slope(contour1), slope(contour2))
    m2inv = 1/m2 # m1inv was not used.
    yi = (m1*(x21-x11-m2inv*y21) + y11)/(1 - m1*m2inv)
    xi = (yi - y21)*m2inv + x21 # (xi, yi) is intersection candidate
    # Bounding box type conditions for intersection candidates
    xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
              amin(x21, x22) < xi, xi <= amax(x21, x22) )
    yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
              amin(y21, y22) < yi, yi <= amax(y21, y22) )
    return xi[aall(xconds)], yi[aall(yconds)]


def identify_pinwheels(re_contours,  im_contours, intersections):
    """
    Locates the pinwheels from the intersection of the real and
    imaginary contours of of polar OR map.
    """
    warning_counter = WarningCounter()
    pinwheels = []
    np.seterrcall(warning_counter)

    for (re_ind, im_ind) in intersections:
        re_contour = re_contours[re_ind]
        im_contour = im_contours[im_ind]
        np.seterr(divide='call', invalid='call')
        x, y = find_intersections(re_contour, im_contour)
        np.seterr(divide='raise', invalid='raise')
        pinwheels += zip(x,y)
    
    warning_counter.warn()        
    return pinwheels
            
            
#==================================#
## Hypercolumn distance estimation #
#==================================#

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    import numpy as np
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

# For featuremapper.analysis

# - Crop to smallest dimension, assume FFT ring in center
#  dim1xdim2 (dim2 > dim1) => dim1xdim1. Warn if dim2-dim1 not even.
# - If dim1 is even, crop anyway and warn.
# - Still have resampling idea.
# - Only this function should fail if scipy not available.

def wavenumber_spectrum(spectrum, reduce_fn=np.mean):
    """
    Bins the power values in the 2D FFT power spectrum as a function
    of wavenumber. Requires square FFT spectra (odd dimension) to
    work.
    """
    dim, _dim = spectrum.shape
    assert dim == _dim, "This approach only supports square FFT spectra"
    assert dim % 2,     "Odd dimensions necessary for properly centered FFT plot"

    # Invert as power_spectrum returns black (low values) for high amplitude
    spectrum = 1 - spectrum
    pixel_bins = range(0, (dim / 2) + 1)
    lower = -(dim / 2); upper = (dim / 2)+1

    # Grid of coordinates relative to central DC component (0,0)
    x,y = np.mgrid[lower:upper, lower:upper]
    flat_pixel_distances= ((x**2 + y**2)**0.5).flatten()
    flat_spectrum = spectrum.flatten()

    # Indices in pixel_bins to which the distances belong
    bin_allocation = np.digitize(flat_pixel_distances, pixel_bins)
    # The bin allocation zipped with actual fft power values
    spectrum_bins = zip(bin_allocation, flat_spectrum)
    grouped_bins = itertools.groupby(sorted(spectrum_bins), lambda x:x[0])
    hist_values = [([sval for (_,sval) in it], bin)
                   for (bin, it) in grouped_bins]
    (power_values, bin_boundaries) = zip(*hist_values)
    averaged_powers = [reduce_fn(power) for power in power_values]
    assert len(bin_boundaries) == len(pixel_bins)
    return averaged_powers


def KaschubeFit(k, a0=0.35, a1=5, a2=1.3, a3=0.15, a4=-0.003, a5=0):
    """
    Fitting function used by Kaschube for finding the hypercolumn
    distance from the Fourier power spectrum. Default values
    correspond to a good starting point for GCAL maps. These values
    should match the init_fit defaults of pinwheel_analysis below.
    a0 => Gaussian height
    a1 => Peak position
    a2 => Gaussian spread (ie. variance)
    a3 => Baseline value (w/o falloff)
    a4 => Linear falloff
    a5 => Quadratic falloff
    """
    exponent = - ((k - a1)**2) / (2 * a2**2)
    return a0 * np.exp(exponent) + a3 + a4*k + a5*np.power(k,2)


def hypercolumn_distance(preference, init_fit=[0.35,3,1.3,0.15,-0.003,0]):
    """
    Estimating the hypercolumn distance by fitting Equation 7 of
    Kaschube et al. 2010 Equation 7 (supplementary material). Returns
    the analysed values in a dictionary.
    """
    fft_spectrum = power_spectrum(preference)
    amplitudes = np.array(azimuthalAverage(1-fft_spectrum))
    #amplitudes = np.array(wavenumber_spectrum(fft_spectrum))
    ks = np.array(range(len(amplitudes)))#[1:]
    amplitudes = amplitudes#[1:]


    err = 100000000000
    valid_fit = False
    for i in xrange(0,20):
        try:
	    init_fit[1] = float(i)
            fit_valst, _ = curve_fit(KaschubeFit, ks, np.array(amplitudes),init_fit, maxfev=10000)
	    fit = dict(zip(['a0','a1','a2','a3','a4','a5'], fit_valst))
	    fit_err =  np.sum(abs(KaschubeFit(ks,fit_valst[0],fit_valst[1],fit_valst[2],fit_valst[3],fit_valst[4],fit_valst[5])-np.array(amplitudes)))
	    print "peak: " , fit['a1'] , "error: " ,fit_err
	    print "height: ", fit['a0']
	    print "width: ", fit['a2']
	    if err > fit_err:
		fit_vals = fit_valst.copy()
		err = fit_err
		valid_fit=True
        except RuntimeError:
	    pass
    
    

    print "picked peak:", fit_vals[1]
    import pylab
    pylab.figure()
    pylab.plot(np.array(amplitudes))
    pylab.plot(KaschubeFit(ks,fit_vals[0],fit_vals[1],fit_vals[2],fit_vals[3],fit_vals[4],fit_vals[5]))

    if valid_fit:
        fit = dict(zip(['a0','a1','a2','a3','a4','a5'], fit_vals))
    
    if (fit['a1'] <= 2) or (fit['a0'] < 0):
       valid_fit = False;
    

    print KaschubeFit(ks[0],fit['a0'],fit['a1'],fit['a2'],fit['a3'],fit['a4'],fit['a5'])
    print KaschubeFit(fit['a1'],fit['a0'],fit['a1'],fit['a2'],fit['a3'],fit['a4'],fit['a5'])
    if KaschubeFit(ks[0],fit['a0'],fit['a1'],fit['a2'],fit['a3'],fit['a4'],fit['a5'])*1.05 >= KaschubeFit(fit['a1'],fit['a0'],fit['a1'],fit['a2'],fit['a3'],fit['a4'],fit['a5']):
       valid_fit = False;	    

    

    kmax_argmax = np.argmax(amplitudes[1:]) + 1
    kmax = fit['a1']+1 if valid_fit else 1000 #float(kmax_argmax)
    #kmax = np.argmax(amplitudes)+1


    print "KMAX", kmax

    # The amplitudes begins with k=0 (DC component), k=1 for one
    # period per map, k=2 for two periods per map etc. The units per
    # hypercolumn is the total number of units across the map divided
    # by kmax. If k <= 1.0, the full map width is reported.
    (dim,_) = fft_spectrum.shape
    units_per_hypercolumn = dim if (kmax <= 1.0) else dim / float(kmax)

    return {'kmax': kmax, # pinwheel_density = pinwheel count/(kmax**2)
            'fit': fit if valid_fit else None,
            'k_delta': kmax - float(kmax_argmax),
            'units_per_hypercolumn':units_per_hypercolumn,
            'amplitudes': amplitudes}
            
            
def pinwheel_analysis(mmap):
    """
    Computes the pinwheels analysis. Fits the hypercolumn distance to
    get the estimated value of kmax to compute the pinwheel density.
    """
    preference = mmap
    #preference = preference * np.pi
    #preference = preference[10:37,10:37]
    #preference = preference[0:47,0:47]
    polar_map = polar_preference(preference)
    contour_info = polarmap_contours(polar_map)
    (re_contours, im_contours, _ ) = contour_info
    pinwheels= identify_pinwheels(*contour_info)
    metadata = hypercolumn_distance(preference)
    metadata.update(rho = (len(pinwheels) / metadata['kmax']**2))
    print "Pinwheel density: ", metadata['rho'], len(pinwheels), metadata['kmax']**2 
    return {'pinwheels':pinwheels,
            're_contours':re_contours,
            'im_contours':im_contours,
            'metadata': metadata}

#=======================================#
# The kernel for the map quality metric #
#=======================================#

try: # 2.7+
    gamma = math.gamma
except:
    import scipy.special as ss
    gamma = ss.gamma

def gamma_dist(x, k, theta=2):
    return (1.0/theta**k)*(1.0/gamma(k)) * x**(k-1) * np.exp(-(x/theta))

def gamma_metric(pwd, k=1.8, mode=math.pi):
    theta = mode / (k -1) # Mode: (k - 1)* theta
    norm = gamma_dist(math.pi, k=k, theta=theta)
    return (1.0/norm)*gamma_dist(pwd, k=k, theta=theta)


###################################################
#                 Visualization                   #
###################################################


def pinwheel_overlay(pinwheels, contours=None, style='wo',linewidth=1,mmap=None):
   """
   Plots the pinwheel locations and optionally the real and imaginary
   pinwheel contours. Designed to be overlayed over an OR map.
   """
   fig = plt.figure(frameon=False)
   fig.patch.set_alpha(0.0)
   ax = plt.subplot(111, aspect='equal', frameon=True)
   ax.patch.set_alpha(0.0)
   plt.hold(True)
    
   plt.imshow(mmap,cmap='hsv',extent=(0, 1.0, 0, 1.0))
   (recontours, imcontours) = contours if contours else ([],[])
   for recontour in recontours:
      plt.plot(recontour[:,0], recontour[:,1],'k',linewidth=linewidth)
   for imcontour in imcontours:
      plt.plot(imcontour[:,0], imcontour[:,1],'w', linewidth=linewidth)

   Xs, Ys = zip(*pinwheels)
   plt.plot(np.array(Xs), np.array(Ys), style)

   plt.xlim((0.0,1.0));         plt.ylim((0.0,1.0))
   ax.xaxis.set_ticks([]);      ax.yaxis.set_ticks([])
   ax.xaxis.set_ticklabels([]); ax.yaxis.set_ticklabels([])
   return fig

def vis(mmap):
    row = pinwheel_analysis(mmap)
    contours = (row['re_contours'], row['im_contours']) 
    contour_fig = pinwheel_overlay(row['pinwheels'], contours=contours,mmap=mmap)
    
