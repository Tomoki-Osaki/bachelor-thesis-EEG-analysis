""" EEG analysis with Python's library MNE: https://mne.tools/stable/index.html """
# pip install mne pyvistaqt ipywidgets ipyevents nibabel darkdetect qdarkdetect 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse
import warnings
warnings.simplefilter('ignore')

subjectID = '036'
exp_condition = 'kazu'
data_path = f"卒業論文/卒論 H201015 データファイル/EEG 実験データ/Mimic_{subjectID}/"
#"C:/Users/ootmo/OneDrive/Documents/卒業論文/卒論 H201015 データファイル/EEG 実験データ/Mimic_{subjectID}/"
path = data_path + 'Mimic_036_kazu_20231107_103715.mff'

# import data folder(.mff) or file(.raw)  
raw = mne.io.read_raw_egi(path, preload=True)
raw.drop_channels(['Vertex Reference']) # when .mff folder is imported 
#raw.drop_channels(['E33']) # when .raw file is imported
raw.set_eeg_reference(projection=True)

# set channel locations: Geodesics Sensor Net HydroCel 32 channels 
montage = mne.channels.make_standard_montage('GSN-HydroCel-32')
raw.set_montage(montage)
# montage.plot()  # 2D
# fig = montage.plot(kind='3d', show=False)  # 3D
# fig = fig.gca().view_init(azim=70, elev=15)
# plt.show()

# regions of interest
ROI = ['E9','E10','E20','E5','E6','E28'] 
ROI_alpha = ['E9','E10','E20'] # alpha waves from the occipital lobe
ROI_mu = ['E5','E6','E28'] # mu waves from the parietal lobe
# channels for EOG
ch_eog = ['E1', 'E2', 'E29', 'E30', 'E31', 'E32'] # channels that are close to eyes

# see raw data
raw.compute_psd(fmax=50).plot(picks='data', exclude='bads')
# raw.plot(start=100, duration=3, n_channels=32)
# plt.show()

# filtering
low_cut = 1 # Hz
hi_cut = 70
raw_filt = raw.copy().filter(low_cut, hi_cut)
raw_filt = raw_filt.notch_filter(60)
#raw_filt.plot()
raw_filt_beforeICA = raw_filt.copy() # to compare before and after ICA

# set up and fit the ICA 
ica = mne.preprocessing.ICA(n_components=None, random_state=97, max_iter=800)
ica.fit(raw_filt)

# remove EOG 
ica.plot_components()
ica.plot_properties(
    raw_filt, 
    picks=range(0, ica.n_components_), 
    psd_args={'fmax': hi_cut}
    )
ica_z_thresh = 1.96
eog_indices, eog_scores = ica.find_bads_eog(
    raw_filt, 
    ch_name=ch_eog, 
    threshold=ica_z_thresh
    )
ica.exclude = eog_indices
ica.plot_scores(eog_scores)
ica.apply(raw_filt)

# compare raw, filtered before-ICA, and filtered after-ICA
raw.plot(start=100, duration=3, n_channels=32) # raw
print('Raw data\n') 
raw_filt_beforeICA.plot(start=100, duration=3, n_channels=32) # filtered before-ICA
print('Filtered before-ICA\n')
raw_filt.plot(start=100, duration=3, n_channels=32) # after-ICA
print('Filtered after-ICA (used in further analysis)\n')

# make event times for epoching
def create_events() -> np.array:
    events = pd.DataFrame(raw._data.T) # raws of ds254 signals
    events = events[32] # 32th raw is a signals of ds254: raw._data  
    index_events = pd.Series(events.index)
    events = pd.DataFrame([events, index_events]).T
    index_events = events.index[events.iloc[:, 0]==0]
    events = events.drop(index_events)
    events = events.drop(events.iloc[:, [0]], axis=1)
    events = events[::2] 
    events['zero'] = 0
    events['events']  = np.tile([1, 2, 3], 10)
    events = np.array(events, dtype='int32')
    
    return events

events = create_events()

event_dict = {'obs1': 1, 'prod': 2, 'obs3': 3}
fig = mne.viz.plot_events(
    events, 
    event_id=event_dict, 
    sfreq=raw.info['sfreq'], 
    first_samp=raw_filt.first_samp
    )

# epoching
reject = {'eeg': 100e-6}
epochs = mne.Epochs(
    raw_filt, 
    events, 
    event_id=event_dict,
    tmin=-2.0, tmax=4.0, 
    picks=ROI, 
    reject=reject, 
    preload=True
    )

conditions = ['obs1', 'prod', 'obs3']
epochs.equalize_event_counts(conditions)

obs1_epochs = epochs['obs1']
prod_epochs = epochs['prod']
obs3_epochs = epochs['obs3']
df_epochs = [obs1_epochs, prod_epochs, obs3_epochs]

# plot data[muV] by condition
for df_epoch, condition in zip(df_epochs, conditions):
    df_epoch.plot_image(picks=ROI)
    print(f'{condition}\n')

# time frequency analysis
baseline = (-2, 0)
kwargs = {'n_cycles': 2, 'return_itc': False, 'freqs': np.arange(8, 14, 1), 'decim': 3}
    
def frequency_analysis(x: str, 
                       power: mne.time_frequency.tfr_morlet, 
                       epochs: mne.Epochs, 
                       plot_power: bool = False, 
                       save_csv: bool = False, 
                       mode: str = 'mean') -> None:
    """ x: obs1 / prod / obs3
        power: mne.time_frequency.tfr_morlet
        epochs: mne.Epochs
        plot_power: if True, plot power using plot_data function
        save_csv: if True, make a dataframe and export it to a csv file
        mode: mean, ratio, logratio, percent, zscore, zlogratio """
    power = mne.time_frequency.tfr_morlet(epochs, **kwargs)
    to_df = power.to_data_frame() # make a data frame that baseline correction is NOT applied yet
    power.apply_baseline(baseline, mode=mode) # calculate relative ERDs to baseline 
    power.plot(ROI_alpha)
    print(f'{x}_alpha: E9, E10, E20\n')
    power.plot(ROI_mu)
    print(f'{x}_mu: E5, E6, E28\n')
    # export to a csv file
    if save_csv == True:
        csv_folder_path = data_path + f'Mimic_{subjectID}_csv'
        if not os.path.exists(csv_folder_path):
            os.mkdir(csv_folder_path)
        to_df['alpha'] = to_df[['E9', 'E10', 'E20']].mean(axis=1)
        to_df['mu'] = to_df[['E5', 'E6', 'E28']].mean(axis=1)    
        csv_file_path = csv_folder_path + f'/Mimic_{subjectID}_{exp_condition}_{x}.csv'
        to_df.to_csv(csv_file_path, index=True)
    # plot power 
    if plot_power == True:
        def plot_data(x: str, df: pd.DataFrame, wave: str) -> None:
            """ x: obs1 / prod / obs3
                df: pd.dataframe
                wave: 'alpha' or 'mu' """
            df['alpha'] = df[['E9','E10', 'E20']].mean(axis=1)
            df['mu'] = df[['E5','E6', 'E28']].mean(axis=1)
            ave_freq = df.groupby(['time'], as_index=False).mean()
            time = ave_freq['time']
            power = ave_freq[wave]
            baseline = power[:167].mean()
            kwargs = {'linestyle': '--', 'linewidth': 2.5}
            sns.relplot(data=ave_freq, x=time, y=power, kind='line', color='blue', alpha=0.5)
            plt.hlines(y=baseline, xmin=-2.0, xmax=4.0, color='red', label='baseline', **kwargs)
            plt.vlines(x=0, ymin=min(power), ymax=max(power), color='green', **kwargs)
            plt.title(f'Power: {x}_{wave}', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=14)
            plt.show()
            
        plot_data(x, to_df, wave='alpha')
        plot_data(x, to_df, wave='mu')

# obs1_power, prod_power, obs3_power are defined for further analysis but not for frequency_analysis
#kwargs = {'n_cycles': 2, 'return_itc': False, 'freqs': np.arange(8, 14, 1), 'decim': 3} # just a reminder
obs1_power = mne.time_frequency.tfr_morlet(obs1_epochs, **kwargs)    
frequency_analysis('obs1', obs1_power, obs1_epochs, plot_power=True, save_csv=False)

prod_power = mne.time_frequency.tfr_morlet(prod_epochs, **kwargs)    
frequency_analysis('prod', prod_power, prod_epochs, plot_power=True, save_csv=False)

obs3_power = mne.time_frequency.tfr_morlet(obs3_epochs, **kwargs)    
frequency_analysis('obs3', obs3_power, obs3_epochs, plot_power=True, save_csv=False)

# The following analyses might not be my interest so far 
# average each condition
obs1_evoked = obs1_epochs.average(picks=ROI)
obs1_evoked_alpha = obs1_epochs.average(picks=ROI_alpha)
obs1_evoked_mu = obs1_epochs.average(picks=ROI_mu)

prod_evoked = prod_epochs.average(picks=ROI)
prod_evoked_alpha = prod_epochs.average(picks=ROI_alpha)
prod_evoked_mu = prod_epochs.average(picks=ROI_mu)

obs3_evoked = obs3_epochs.average(picks=ROI)
obs3_evoked_alpha = obs3_epochs.average(picks=ROI_alpha)
obs3_evoked_mu = obs3_epochs.average(picks=ROI_mu)

def plot_evoked(x, data_evoked, data_evoked_alpha, data_evoked_mu):
    data_evoked.plot()
    print(f'{x}: E9, E10, E20, E5, E6, E28')
    data_evoked_alpha.plot()
    print(f'{x}_alpha: E9, E10, E20')
    data_evoked_mu.plot()
    print(f'{x}_mu: E5, E6, E28')

plot_evoked('obs1', obs1_evoked, obs1_evoked_alpha, obs1_evoked_mu)    
plot_evoked('prod', prod_evoked, prod_evoked_alpha, prod_evoked_mu)
plot_evoked('obs3', obs3_evoked, obs3_evoked_alpha, obs3_evoked_mu)

# plot each condition's averaged data
kwargs = {'legend': 'upper left', 'show_sensors': 'upper right'}
mne.viz.plot_compare_evokeds(
    {'obs1': obs1_evoked_alpha, 
     'prod': prod_evoked_alpha, 
     'obs3': obs3_evoked_alpha}, 
    **kwargs
    )
print('alpha: E9, E10, E20')

mne.viz.plot_compare_evokeds(
    {'obs1': obs1_evoked_mu, 'prod': prod_evoked_mu, 'obs3': obs3_evoked_mu}, **kwargs)
print('mu: E5, E6, E28')

# plot topography 
def plot_topography(x, data_evoked_alpha, data_evoked_mu):
    times = [-2, -1, 0, 1, 2, 3, 4]
    data_evoked_alpha.plot_joint()
    data_evoked_alpha.plot_topomap(times=times, ch_type='eeg')
    print(f'{x}_alpha\n')
    data_evoked_mu.plot_joint()
    data_evoked_mu.plot_topomap(times=times, ch_type='eeg')
    print(f'{x}_mu\n')

plot_topography('obs1', obs1_evoked_alpha, obs1_evoked_mu)
plot_topography('prod', prod_evoked_alpha, prod_evoked_mu)
plot_topography('obs3', obs3_evoked_alpha, obs3_evoked_mu)

# # EEG forward operator with a template MRI for an adult
# # Download fsaverage files 
# fs_dir = fetch_fsaverage(verbose=True) # fs:FreeSurfer
# subjects_dir = os.path.dirname(fs_dir)

# # The files live in 'C:\Users\ootmo\mne_data\MNE-fsaverage-data\fsaverage'
# subject = 'fsaverage'
# trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
# src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') # src stands for source space
# bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif') # bem stands for boundary element method

# mne.viz.set_3d_options(multi_samples=1, antialias=False)
# mne.viz.set_3d_backend('pyvistaqt')
# mne.viz.plot_alignment(
#     raw.info, src=src, eeg=['original', 'projected'],
#     trans=trans, show_axes=True, mri_fiducials=True,
#     dig='fiducials')

# fwd = mne.make_forward_solution(
#     raw.info, trans=trans, src=src, bem=bem,
#     meg=False, eeg=True, mindist=5.0, n_jobs=None)
# print(fwd)

# # Source localization with dSPM
# kwargs = {'tmax': 0, 'method': ['shrunk', 'empirical'], 'rank': None, 'verbose': True}
# noise_cov_obs1 = mne.compute_covariance(obs1_epochs, **kwargs)
# fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov_obs1, raw.info)

# noise_cov_prod = mne.compute_covariance(prod_epochs, **kwargs)
# fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov_prod, raw.info)

# noise_cov_obs3 = mne.compute_covariance(obs3_epochs, **kwargs)
# fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov_obs3, raw.info)

# kwargs = {'loose': 0.2, 'depth': 0.8}
# obs1_inverse_operator = make_inverse_operator(obs1_evoked.info, fwd, noise_cov_obs1, **kwargs)
# prod_inverse_operator = make_inverse_operator(prod_evoked.info, fwd, noise_cov_prod, **kwargs)
# obs3_inverse_operator = make_inverse_operator(obs3_evoked.info, fwd, noise_cov_obs3, **kwargs)

# snr = 3.0 # signal to noise ratio
# lambda2 = 1.0 / snr**2
# kwargs = {'initial_time': 0.08, 
#           'hemi': 'both', 
#           'subjects_dir': subjects_dir, 
#           'size': (600, 600), 
#           'clim': {'kind': 'percent', 'lims': [90, 95, 99]}, 
#           'smoothing_steps': 7}

# def plot_brain_activity(df_evoked, df_inverse_operator):
#     df_stc = abs(apply_inverse(df_evoked, df_inverse_operator, lambda2, 'dSPM', verbose=True))
#     df_brain = df_stc.plot(figure=1, **kwargs)
#     df_brain.add_text(0.1, 0.9, 'dSPM', 'title', font_size=14)    
    
# # DO NOT RUN all the followings at one time
# plot_brain_activity(obs1_evoked, obs1_inverse_operator)
# plot_brain_activity(prod_evoked, prod_inverse_operator)
# plot_brain_activity(obs3_evoked, obs3_inverse_operator)
