import pickle
import os
import stat
import numpy
import sys
from pinwheel_analysis import *
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append('/home/jan/projects/topographica/')
import topo
from topo.command.analysis import save_plotgroup
from topo.command import load_snapshot

if False:
    f = open('/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_EI_II/exc_inh_strength=3.4_inh_inh_strength=0.3/results.pickle')
    d = pickle.load(f);f.close()
    #mmap = d['orprefmap'][25:-24,25:-24]
    mmap = d['orprefmap'][2:-3,2:-3]
    rho = pinwheel_analysis(mmap)['metadata']['rho']
    print "RHO:", rho
    metric = gamma_metric(rho,k=5.0)
    print "METRIC:",metric
    vis(mmap)
    print "MEAN:", numpy.mean(mmap)
    pylab.figure();pylab.imshow(mmap,interpolation='none',cmap='hsv')
    pylab.figure();pylab.imshow(abs(fftshift(fft2(mmap-0.5))),interpolation='none',cmap='gray')
    pylab.show()
    0/0


def figure2():
    dirr = '/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_SHORTRANGE'
    rhos = {}
    
    X = []
    Y = []
    qual = []
    
    if False:
        for a in os.listdir(dirr):
            b = os.path.join(dirr,a);

            if not stat.S_ISDIR(os.stat(b).st_mode):
                continue

            # load result file 
            f = open(os.path.join(b,'results.pickle'))
            d = pickle.load(f)
            
            # lets find out the pinwheel density
            
            X.append(d['lat_strength_ratio'])
            Y.append(d['exc_inh_ratio'])

	    #mmap = d['orprefmap'][25:-24,25:-24]
	    mmap = d['orprefmap'][2:-3,2:-3]
	    
            rho = pinwheel_analysis(mmap)['metadata']['rho']
            metric = gamma_metric(rho,k=10.0)
            sel = numpy.mean(numpy.mean(mmap))
            print a, " ", str(rho) , " " , str(metric) , "SEL ", str(sel)

            qual.append(metric)
            
        data = np.histogram2d(Y, X, bins=[len(np.unique(Y)),len(np.unique(X))], weights=qual)
        pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
        im = pylab.imshow(data[0],interpolation='none',cmap='gray')#,vmin=0.3)
        pylab.colorbar(im,fraction=0.046, pad=0.04)
        pylab.yticks([0,len(np.unique(Y))-1],[data[1][0],data[1][-1]])
        pylab.xticks([0,len(np.unique(X))-1],[data[2][0],data[2][-1]])
        pylab.savefig('figure2.png',dpi=600)
    

    if True:
        X = []
	Y = []
        qual = []


        for a in os.listdir(dirr):
            b = os.path.join(dirr,a);

            if not stat.S_ISDIR(os.stat(b).st_mode):
                continue
	    
	    f = open(os.path.join(b,'results.pickle'))
            d = pickle.load(f)
            
	    X.append(d['lat_strength_ratio'])
            Y.append(d['exc_inh_ratio'])

            # load result file 
            load_snapshot(os.path.join(b,'snapshot.typ'))
    	    qual.append(numpy.mean(topo.sim["V1"].sheet_views["OrientationSelectivity"].view()[0]))
	    print qual[-1]

	data = np.histogram2d(Y, X, bins=[len(np.unique(Y)),len(np.unique(X))], weights=qual)
        pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
        im = pylab.imshow(data[0],interpolation='none',cmap='gray')#,vmin=0.3)
        pylab.colorbar(im,fraction=0.046, pad=0.04)
        pylab.yticks([0,len(np.unique(Y))-1],[data[1][0],data[1][-1]])
        pylab.xticks([0,len(np.unique(X))-1],[data[2][0],data[2][-1]])
        pylab.savefig('figure_sel.png',dpi=600)


    if False:
       f = open('/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_SHORTRANGE/lat_strength_ratio=2.2_exc_inh_ratio=0.85/results.pickle') 
       d = pickle.load(f)
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(d['orprefmap'][61:-60,61:-60],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure1/generated_data/or_map3.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(d['orprefmap'][61:-60,61:-60]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure1/generated_data/fft_map3.png',  pad_inches=0)
              
       load_snapshot('/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_SHORTRANGE/lat_strength_ratio=2.2_exc_inh_ratio=0.85/snapshot.typ')
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[0],density=1.5,saver_params={'filename_prefix' : '3'})
       
       f = open('/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_SHORTRANGE/lat_strength_ratio=2.6_exc_inh_ratio=0.8/results.pickle') 
       d = pickle.load(f)
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(d['orprefmap'][61:-60,61:-60],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure1/generated_data/or_map2.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(d['orprefmap'][61:-60,61:-60]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure1/generated_data/fft_map2.png',  pad_inches=0)


       load_snapshot('/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_SHORTRANGE/lat_strength_ratio=2.6_exc_inh_ratio=0.8/snapshot.typ')
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[0],density=1.5,saver_params={'filename_prefix' : '2'})
       
       f = open('/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_SHORTRANGE/lat_strength_ratio=2.8_exc_inh_ratio=0.65/results.pickle') 
       d = pickle.load(f)
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(d['orprefmap'][61:-60,61:-60],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure1/generated_data/or_map1.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(d['orprefmap'][61:-60,61:-60]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure1/generated_data/fft_map1.png',  pad_inches=0)
     
       load_snapshot('/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_SHORTRANGE/lat_strength_ratio=2.8_exc_inh_ratio=0.65/results.pickle')
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[0],density=1.5,saver_params={'filename_prefix' : '1'})

    if False:
       load_snapshot('/home/jan/projects/topographica/GCAL_EI/a-p_exc_strength=3.3_-p_inh_strength=2.805/snapshot.typ')
       mmap = topo.sim["V1"].sheet_views["OrientationPreference"].view()[0]

       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(mmap[61:-60,61:-60],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure1/generated_data/or_map4.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(mmap[61:-60,61:-60]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure1/generated_data/fft_map4.png',  pad_inches=0)
     
       
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[0],density=1.5,saver_params={'filename_prefix' : '4'})
        
def figure3():
    dirr = '/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_EI_II'
    rhos = {}
    
    X = []
    Y = []
    qual = []

    if True:
        for a in os.listdir(dirr):
            b = os.path.join(dirr,a);

            if not stat.S_ISDIR(os.stat(b).st_mode):
                continue

            # load result file 
            f = open(os.path.join(b,'results.pickle'))
            d = pickle.load(f)
            
            # lets find out the pinwheel density
            
            X.append(abs(d['exc_inh_strength']))
            Y.append(abs(d['inh_inh_strength']))
            rho = pinwheel_analysis(d['orprefmap'][2:-3,2:-3])['metadata']['rho']
            metric = gamma_metric(rho,k=10.0)
            sel = numpy.mean(numpy.mean(d['orselmap'][2:-3,2:-3]))
            print a, " ", str(rho) , " " , str(metric) , "SEL ", str(sel)

            qual.append(metric)
            
        data = np.histogram2d(Y, X, bins=[len(np.unique(Y)),len(np.unique(X))], weights=qual)
        pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
	ax = plt.gca()
	im = pylab.imshow(data[0],interpolation='none',cmap='gray')#,vmin=0.3)
        pylab.yticks([0,len(np.unique(Y))-1],[data[1][0],data[1][-1]])
        pylab.xticks([0,len(np.unique(X))-1],[data[2][0],data[2][-1]])
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im,cax=cax)
        pylab.savefig('figure3.png',dpi=600)
    
    if False:

       fname= '/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_EI_II/exc_inh_strength=3.5_inh_inh_strength=0.2'
       f = open(fname+'/results.pickle') 
       d = pickle.load(f)
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(d['orprefmap'][0:-1,0:-1],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure2/generated_data/or_map1.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(d['orprefmap'][0:-1,0:-1]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure2/generated_data/fft_map1.png',  pad_inches=0)
              
       load_snapshot(fname+'/snapshot.typ')
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[0],density=1.5,saver_params={'filename_prefix' : '1'})


       fname= '/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_EI_II/exc_inh_strength=3.7_inh_inh_strength=0.4'
       f = open(fname+'/results.pickle') 
       d = pickle.load(f)
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(d['orprefmap'][0:-1,0:-1],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure2/generated_data/or_map2.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(d['orprefmap'][0:-1,0:-1]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure2/generated_data/fft_map2.png',  pad_inches=0)
              
       load_snapshot(fname+'/snapshot.typ')
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[0],density=1.5,saver_params={'filename_prefix' : '2'})
    
       fname= '/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_EI_II/exc_inh_strength=3.7_inh_inh_strength=1.0'
       f = open(fname+'/results.pickle') 
       d = pickle.load(f)
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(d['orprefmap'][0:-1,0:-1],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure2/generated_data/or_map3.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(d['orprefmap'][0:-1,0:-1]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure2/generated_data/fft_map3.png',  pad_inches=0)
              
       load_snapshot(fname+'/snapshot.typ')
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[0],density=1.5,saver_params={'filename_prefix' : '3'})


def figure4():
    dirr = '/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_EI_II_LONGE'
    rhos = {}
    
    X = []
    Y = []
    qual = []

    if False:
        for a in os.listdir(dirr):
            b = os.path.join(dirr,a);

            if not stat.S_ISDIR(os.stat(b).st_mode):
                continue

            # load result file 
            f = open(os.path.join(b,'results.pickle'))
            d = pickle.load(f)
            
            # lets find out the pinwheel density
            
            X.append(abs(d['exc_inh_strength']))
            Y.append(abs(d['exc_short_long_ratio']))
	    
	    mmap = d['orprefmap'][2:-3,2:-3]

            rho = pinwheel_analysis(mmap)['metadata']['rho']
            metric = gamma_metric(rho,k=10.0)
            sel = numpy.mean(numpy.mean(mmap))
            print a, " ", str(rho) , " " , str(metric) , "SEL ", str(sel)

            qual.append(metric)
	
	print X
	print Y
	print qual
	pylab.figure()
	pylab.scatter(Y,X,s=numpy.array(qual)*50.0)

        data = np.histogram2d(Y, X, bins=[len(np.unique(Y)),len(np.unique(X))], weights=qual)
        pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
	ax = plt.gca()
	im = pylab.imshow(data[0],interpolation='none',cmap='gray')#,vmin=0.3)
        pylab.yticks([0,len(np.unique(Y))-1],[data[1][0],data[1][-1]])
        pylab.xticks([0,len(np.unique(X))-1],[data[2][0],data[2][-1]])
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im,cax=cax)
        pylab.savefig('figure4.png',dpi=600)
    
    if True:

       fname= '/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_EI_II_LONGE/a-p_exc_strength=9_-p_exc_inh_strength=9.7_-p_exc_short_long_ratio=0.3_-p_cortex_exc_target_activity=0.003'
       f = open(fname+'/results.pickle') 
       d = pickle.load(f)
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(d['orprefmap'][0:-1,0:-1],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure3/generated_data/or_map1.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(d['orprefmap'][0:-1,0:-1]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure3/generated_data/fft_map1.png',  pad_inches=0)
              
       load_snapshot(fname+'/snapshot.typ')
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[2],density=6.0,saver_params={'filename_prefix' : '1'})

       fname= '/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_EI_II_LONGE/a-p_exc_strength=9_-p_exc_inh_strength=10.6_-p_exc_short_long_ratio=0.5_-p_cortex_exc_target_activity=0.003'
       f = open(fname+'/results.pickle') 
       d = pickle.load(f)
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(d['orprefmap'][0:-1,0:-1],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure3/generated_data/or_map2.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(d['orprefmap'][0:-1,0:-1]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure3/generated_data/fft_map2.png',  pad_inches=0)
              
       load_snapshot(fname+'/snapshot.typ')
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[2],density=6.0,saver_params={'filename_prefix' : '2'})
    
       fname= '/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_EI_II_LONGE/a-p_exc_strength=9_-p_exc_inh_strength=9.4_-p_exc_short_long_ratio=0.6_-p_cortex_exc_target_activity=0.003'
       f = open(fname+'/results.pickle') 
       d = pickle.load(f)
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(d['orprefmap'][0:-1,0:-1],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure3/generated_data/or_map3.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(d['orprefmap'][0:-1,0:-1]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure3/generated_data/fft_map3.png',  pad_inches=0)
              
       load_snapshot(fname+'/snapshot.typ')
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[2],density=6.0,saver_params={'filename_prefix' : '3'})

       fname= '/home/jan/Doc/Papers/fast_inh_paper/DATA/GCAL_EI_II_LONGE/a-p_exc_strength=9_-p_exc_inh_strength=10.6_-p_exc_short_long_ratio=0.8_-p_cortex_exc_target_activity=0.003'
       f = open(fname+'/results.pickle') 
       d = pickle.load(f)
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(d['orprefmap'][0:-1,0:-1],interpolation='none',cmap='hsv',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure3/generated_data/or_map4.png',  pad_inches=0)
       
       fig = pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
       ax = plt.Axes(fig, [0., 0., 1., 1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(power_spectrum(d['orprefmap'][0:-1,0:-1]),interpolation='none',cmap='gray',aspect='normal')
       pylab.savefig('/home/jan/Doc/Papers/fast_inh_paper/SVG/Figure3/generated_data/fft_map4.png',  pad_inches=0)
              
       load_snapshot(fname+'/snapshot.typ')
       save_plotgroup("Projection",projection=topo.sim["V1"].projections().values()[2],density=6.0,saver_params={'filename_prefix' : '4'})


def figure5():
    dirr = '/home/jan/Doc/Papers/fast_inh_paper/DATA/LISSOM_SHORTRANGE_EI'
    rhos = {}
    
    X = []
    Y = []
    qual = []
    rho = []
    # lets calculate the orientation preference and selectivity
    num_orientation = 8
    angles = [2*numpy.pi/num_orientation*i for i in xrange(num_orientation)] 
    angles_as_complex_numbers = numpy.cos(angles) + numpy.sin(angles) * 1j
    maps = {}
    if True:
        for a in os.listdir(dirr):
            b = os.path.join(dirr,a);

            if not stat.S_ISDIR(os.stat(b).st_mode):
                continue

            # load result file 
	    print b
            f = open(os.path.join(b,'freq=4_scale=10_resp.pickle'))
            d = pickle.load(f)
            resp = numpy.max(d['V1'],axis=1)
            polar_resp = resp * numpy.array(angles_as_complex_numbers)[:,numpy.newaxis]
            polar_mean_resp = numpy.mean(polar_resp,axis=0)
            mmap = ((numpy.angle(polar_mean_resp) + 4*numpy.pi) % (numpy.pi*2)) / (numpy.pi*2)
            sheet_size = numpy.sqrt(len(mmap))
            mmap = numpy.resize(mmap,(sheet_size,sheet_size))
            # lets find out the pinwheel density
            mmap = mmap[2:-3,2:-3]
            X.append(abs(float(a.split('_')[5])))

	    if X[-1] == 0.00019:
	        rho = pinwheel_analysis(mmap)['metadata']['rho']
	        print "RHO:", rho
	        metric = gamma_metric(rho,k=10.0)
	        print "METRIC:",metric
	        vis(mmap)
	        pylab.figure();pylab.imshow(mmap,interpolation='none',cmap='hsv')
	        pylab.figure();pylab.imshow(abs(fftshift(fft2(mmap-0.5))),interpolation='none',cmap='gray')
	        pylab.show()
	    #else:
	#	continue

            maps[X[-1]] = mmap
            rho = pinwheel_analysis(mmap)['metadata']['rho']
            metric = gamma_metric(rho,k=10.0)
            sel = numpy.mean(numpy.mean(mmap))
            print a, " ", str(rho) , " " , str(metric) , "SEL ", str(sel)
            qual.append(metric)


 

        data = np.histogram(X, bins=len(np.unique(X)), weights=qual)
        pylab.figure(dpi=600,facecolor='w',figsize=(5,5))
        ax = plt.gca()
        im = pylab.imshow([data[0]],interpolation='none',cmap='gray',vmin=0.0,vmax=1.0)
        pylab.xticks([0,len(np.unique(X))-1],[sorted(X)[0],sorted(X)[-1]])
        pylab.yticks([])
        divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #pylab.colorbar(im,cax=cax)
        #pylab.colorbar()
        pylab.savefig('figure5.png',dpi=600)


def a():
    dirr = '/home/jan/projects/topographica/GCAL_EI/'
    rhos = {}
    
    X = []
    Y = []
    qual = []

    if True:
        for a in os.listdir(dirr):
            b = os.path.join(dirr,a);

            if not stat.S_ISDIR(os.stat(b).st_mode):
                continue

            # load result file 
            f = open(os.path.join(b,'results.pickle'))
            d = pickle.load(f)
	    f.close()
            load_snapshot(os.path.join(b,'snapshot.typ'))


	    f = open(os.path.join(b,'results.pickle'),'w')
	    if topo.sim["V1"].projections()["EtoELong"].strength != 0:
		    d['exc_short_long_ratio'] = 1 - 1 / (topo.sim["V1"].projections()["EtoE"].strength / topo.sim["V1"].projections()["EtoELong"].strength+1)
	    else:
		    d['exc_short_long_ratio'] = 1.0
	    print d['exc_short_long_ratio']
	    pickle.dump(d,f)
            

#a()
figure2()
#figure3()
#figure4()
#a()
#figure5()

