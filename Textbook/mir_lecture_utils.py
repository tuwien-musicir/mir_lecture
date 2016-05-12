# numerical processing and scientific libraries

# Numpy is the fundamental package for scientific computing with Python. It implements a wide range of fast 
# and powerful algebraic functions.
import numpy as np

# Scikit-Learn is a powerful machine learning package for Python built on numpy and Scientific Python (Scipy).
import scipy

import pandas as pd

# signal processing
from scipy.io                     import wavfile
from scipy                        import stats, signal
from scipy.fftpack                import fft

from scipy.signal                 import lfilter, hamming
from scipy.fftpack.realtransforms import dct
from scipy.io.arff                import loadarff

# Talkbox, a set of python modules for speech/signal processing
from scikits.talkbox              import segment_axis
from scikits.talkbox.features     import mfcc

# general purpose
import collections

# plotting
from   numpy.lib                  import stride_tricks
import matplotlib        as mpl
import matplotlib.pyplot as plt
import seaborn           as sns

from IPython.display              import HTML
from base64                       import b64encode

# Classification and evaluation
from sklearn.preprocessing        import StandardScaler
from sklearn                      import svm
from sklearn.cross_validation     import StratifiedKFold, ShuffleSplit, cross_val_score
from sklearn.naive_bayes          import GaussianNB
from sklearn.neighbors            import KNeighborsClassifier
from sklearn.ensemble             import RandomForestClassifier
from sklearn.metrics              import classification_report, confusion_matrix

import librosa as lr

ABBRIVATIONS = {}

# features
ABBRIVATIONS["zcr"] = "Zero Crossing Rate"
ABBRIVATIONS["rms"] = "Root Mean Square"
ABBRIVATIONS["sc"]  = "Spectral Centroid"
ABBRIVATIONS["sf"]  = "Spectral Flux"
ABBRIVATIONS["sr"]  = "Spectral Rolloff"

# aggregations
ABBRIVATIONS["var"] = "Variance"
ABBRIVATIONS["std"] = "Standard Deviation"
ABBRIVATIONS["mean"] = "Average"

PLOT_WIDTH  = 15
PLOT_HEIGHT = 3.5


def normalize_wav(wavedata,samplewidth=2):

    # samplewidth in byte (i.e.: 1 = 8bit, 2 = 16bit, 3 = 24bit, 4 = 32bit)
    divisor  = 2**(8*samplewidth)/2
    wavedata = wavedata / float(divisor)
    return (wavedata)

def show_mono_waveform(sound_files, genre):

    fig = plt.figure(num=None, figsize=(PLOT_WIDTH, 4), dpi=72, facecolor='w', edgecolor='k')
    lr.display.waveplot(normalize_wav(sound_files[genre]["wavedata"]), sr=sound_files[genre]["samplerate"], alpha=0.75)
    plt.title('Waveform')
    plt.tight_layout()


def show_stereo_waveform(sound_files, genre):

    fig = plt.figure(num=None, figsize=(PLOT_WIDTH, 5), dpi=72, facecolor='w', edgecolor='k');

    channel_1 = fig.add_subplot(211);
    channel_1.set_ylabel('Channel 1');
    lr.display.waveplot(normalize_wav(sound_files[genre]["wavedata"][:,0]), sr=sound_files[genre]["samplerate"], alpha=0.75);


    channel_2 = fig.add_subplot(212);
    channel_2.set_ylabel('Channel 2');
    lr.display.waveplot(normalize_wav(sound_files[genre]["wavedata"][:,1]), sr=sound_files[genre]["samplerate"], alpha=0.75);
    #plt.title('Waveform')

    plt.show();
    plt.clf();


    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):

    win = window(frameSize)

    result = lr.core.stft(sig, n_fft=frameSize, window=win)


    return result


""" plot spectrogram"""
def plotstft(sound_files, genre, binsize=2**10, plotpath=None, colormap="jet", ax=None, fig=None):

    win = np.hanning(binsize)

    wavedata = sound_files[genre]["wavedata"]
    samplerate = sound_files[genre]["samplerate"] * 2

    if len(wavedata.shape) > 1:
        wavedata = wavedata[:,0]

    D = lr.core.stft(wavedata, n_fft=binsize, window=win)

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(PLOT_WIDTH, 3.5))
    lr.display.specshow(lr.logamplitude(np.abs(D)**2, ref_power=np.max), sr=samplerate, y_axis='log', x_axis='time')#, cmap = sns.cubehelix_palette(light=1, as_cmap=True))
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()


def list_audio_samples(sound_files):

	src = ""

	for genre in sound_files.keys():
	
		src += "<b>" + genre + "</b><br><br>"
		src += "<object width='600' height='90'><param name='movie' value='http://freemusicarchive.org/swf/trackplayer.swf'/><param name='flashvars' value='track=http://freemusicarchive.org/services/playlists/embed/track/{0}.xml'/><param name='allowscriptaccess' value='sameDomain'/><embed type='application/x-shockwave-flash' src='http://freemusicarchive.org/swf/trackplayer.swf' width='600' height='50' flashvars='track=http://freemusicarchive.org/services/playlists/embed/track/{0}.xml' allowscriptaccess='sameDomain' /></object><br><br>".format(sound_files[genre]["online_id"])
		
	return HTML(src) 

def play_sample(sound_files, genre):

	src = "<object width='600' height='90'><param name='movie' value='http://freemusicarchive.org/swf/trackplayer.swf'/><param name='flashvars' value='track=http://freemusicarchive.org/services/playlists/embed/track/{0}.xml'/><param name='allowscriptaccess' value='sameDomain'/><embed type='application/x-shockwave-flash' src='http://freemusicarchive.org/swf/trackplayer.swf' width='600' height='50' flashvars='track=http://freemusicarchive.org/services/playlists/embed/track/{0}.xml' allowscriptaccess='sameDomain' /></object><br><br>".format(sound_files[genre]["online_id"])
		
	return HTML(src) 
	

def plot_compairison(data, feature, aggregators):
    
    width = 0.35

    features = {}

    for aggr_name in aggregators:
        
        features[aggr_name] = []
        
        for genre in data.keys():

            if aggr_name == "mean":
                features[aggr_name].append(np.mean(data[genre][feature]))
            elif aggr_name == "std":
                features[aggr_name].append(np.std(data[genre][feature]))
            elif aggr_name == "var":
                features[aggr_name].append(np.var(data[genre][feature]))
            elif aggr_name == "median":
                features[aggr_name].append(np.median(data[genre][feature]))
            elif aggr_name == "min":
                features[aggr_name].append(np.min(data[genre][feature]))
            elif aggr_name == "max":
                features[aggr_name].append(np.max(data[genre][feature]))
    
    fig, ax = plt.subplots()
    ind     = np.arange(len(features[aggregators[0]]))
    rects1  = ax.bar(ind, features[aggregators[0]], 0.7, color='b')
    ax.set_xticklabels( data.keys() )
    ax.set_xticks(ind+width)
    ax.set_ylabel(ABBRIVATIONS[aggregators[0]])
    ax.set_title("{0} Results".format(ABBRIVATIONS[feature]))
    
    
    if len(aggregators) == 2:
    
        ax2 = ax.twinx()
        ax2.set_ylabel(ABBRIVATIONS[aggregators[1]])
        rects2 = ax2.bar(ind+width, features[aggregators[1]], width, color='y')
        ax.legend( (rects1[0], rects2[0]), (ABBRIVATIONS[aggregators[0]], ABBRIVATIONS[aggregators[1]]) )
    
    plt.show()  

def show_feature_superimposed(sound_files, genre, feature, binsize=1024, plot_on="waveform"):

    wavedata     = sound_files[genre]["wavedata"]
    samplerate   = sound_files[genre]["samplerate"]
    timestamps   = sound_files[genre]["%s_timestamp" % (feature)]
    feature_data = sound_files[genre][feature]

    #TODO debug scale and remove if possible

    if feature == "sc":
        scale = 250.0
    elif feature == "zcr":
        scale = 1000.0
    elif feature == "rms":
        scale = 1000.0
    elif feature == "sr":
        scale = 250.0
    elif feature == "sf":
        scale = 250.0


    # plot feature-data
    scaled_fd_y = timestamps * scale


    win = np.hanning(binsize)

    if len(wavedata.shape) > 1:
        wavedata = wavedata[:,0]

    D = lr.core.stft(wavedata, n_fft=binsize, window=win)

    fig, ax = plt.subplots(2, 1, sharex=False, figsize=(PLOT_WIDTH, 7), sharey=True)

    # show spectrogram
    plt.subplot(2, 1, 1)
    lr.display.specshow(lr.logamplitude(np.abs(D)**2, ref_power=np.max), sr=samplerate*2, y_axis='log', x_axis='time')

    if plot_on == "spectrogram":
        scaled_fd_x = feature_data
        _ = plt.plot(scaled_fd_y, scaled_fd_x, color='r', linewidth=1);
        #ax = plt.gca().set_yscale("log")

    # show waveform
    plt.subplot(2, 1, 2);
    lr.display.waveplot(normalize_wav(wavedata), sr=samplerate, alpha=0.75);

    if plot_on == "waveform":
        scaled_fd_x = (feature_data / np.max(feature_data));
        _ = plt.plot(scaled_fd_y, scaled_fd_x, color='r', linewidth=1);

        ax = plt.gca()
        ax.axhline(y=0,c="green",linewidth=3,zorder=0)

    plt.tight_layout();

    plt.show();
    plt.clf();

def load_features_from_arff(path):

    data, meta = loadarff(path)
    features = pd.DataFrame(data, columns=meta)
    features[features.columns[:-1]] = StandardScaler().fit_transform(features[features.columns[:-1]])

    return features

def nextpow2(num):
    n = 2 
    i = 1
    while n < num:
        n *= 2 
        i += 1
    return i

def periodogram(x,win,Fs=None,nfft=1024):
        
    if Fs == None:
        Fs = 2 * np.pi
   
    U  = np.dot(win.conj().transpose(), win) # compensates for the power of the window.
    Xx = fft((x * win),nfft) # verified
    P  = Xx*np.conjugate(Xx)/U
    
    # Compute the 1-sided or 2-sided PSD [Power/freq] or mean-square [Power].
    # Also, compute the corresponding freq vector & freq units.
    
    # Generate the one-sided spectrum [Power] if so wanted
    if nfft % 2 != 0:
        select = np.arange((nfft+1)/2)  # ODD
        P_unscaled = P[select,:] # Take only [0,pi] or [0,pi)
        P[1:-1] = P[1:-1] * 2 # Only DC is a unique point and doesn't get doubled
    else:
        select = np.arange(nfft/2+1);    # EVEN
        P = P[select]         # Take only [0,pi] or [0,pi) # todo remove?
        P[1:-2] = P[1:-2] * 2
    
    P = P / (2* np.pi)

    return P

def map_labels_to_numbers(eval_data):
    
    for df_name in eval_data.keys():

        # create label mapping
        label_mapping = {}
        num_to_label  = []

        i = 0
        for l in set(eval_data[df_name]["labels"]):
            label_mapping[l] = i
            num_to_label.append(l)
            i += 1

        eval_data[df_name]["label_mapping"] = label_mapping
        eval_data[df_name]["num_to_label"] = num_to_label

        mapped_labels = []

        for i in range(eval_data[df_name]["labels"].shape[0]):
            #print label_mapping[ls[i]]
            mapped_labels.append(label_mapping[eval_data[df_name]["labels"][i]])

        #transformed_label_space.append(mapped_labels)

        eval_data[df_name]["num_labels"] = np.asarray(mapped_labels)
    
#styles = "<style>div.cell{ width:900px; margin-left:0%; margin-right:auto;} </style>"
#HTML(styles)

