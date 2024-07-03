import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15
sns.set_style('whitegrid')

# importing data and process
subjects = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', 
            '011', '012', '013', '014', '015', '016', '017', '018', '019', '020',
            '021', '022', '025', '026', '027', '028', '029', '030', '031', '032',
            '034', '036']

colnames = ['C1_alpha', 'C1_mu', 'Cp_alpha', 'Cp_mu', 'C3_alpha', 'C3_mu',
            'M1_alpha', 'M1_mu', 'Mp_alpha', 'Mp_mu', 'M3_alpha', 'M3_mu']

def import_data(sigma_threshold: int = None, 
                ratio_to_baseline: bool = False
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ sigma_threshold: 2 or 3 to remove outliers
        ratio_to_baseline: if True, calculate ratio to baseline """
    dfdict = {'conditions': colnames}
    mean_list = []
    df_individual = {}
    df_tsa = pd.DataFrame(index=range(501), columns=colnames)
    df_tsa = df_tsa.fillna(0)
    df_tsa.insert(0, 'time', 0)
    
    for ID in subjects:
        path = f'C:/Users/ootmo/OneDrive/Documents/Mimic 2023 mff folders/Mimic_{ID}/Mimic_{ID}_csv/'
        C1 = pd.read_csv(path + f'Mimic_{ID}_kazu_obs1.csv').groupby(['time'], as_index=False).mean()
        Cp = pd.read_csv(path + f'Mimic_{ID}_kazu_prod.csv').groupby(['time'], as_index=False).mean()
        C3 = pd.read_csv(path + f'Mimic_{ID}_kazu_obs2.csv').groupby(['time'], as_index=False).mean()
        M1 = pd.read_csv(path + f'Mimic_{ID}_imi_obs1.csv').groupby(['time'], as_index=False).mean()
        Mp = pd.read_csv(path + f'Mimic_{ID}_imi_prod.csv').groupby(['time'], as_index=False).mean()
        M3 = pd.read_csv(path + f'Mimic_{ID}_imi_obs2.csv').groupby(['time'], as_index=False).mean()    
        dfs = [C1, Cp, C3, M1, Mp, M3]

        for df in dfs:
            if sigma_threshold:
                df['alpha'] = df['alpha'][abs(df['alpha'] - df['alpha'].mean() < df['alpha'].std() * sigma_threshold)]
                df['mu'] = df['mu'][abs(df['mu'] - df['mu'].mean() < df['mu'].std() * sigma_threshold)]
            if ratio_to_baseline == True: 
                # transform values into ratio to baseline
                baseline = df['alpha'][:167].mean()
                df['alpha'] = (df['alpha'] - baseline) / baseline
                baseline = df['mu'][:167].mean()
                df['mu'] = (df['mu'] - baseline) / baseline
            else: pass
        dfs_waves = [C1['alpha'], C1['mu'], Cp['alpha'], Cp['mu'], C3['alpha'], C3['mu'], 
                     M1['alpha'], M1['mu'], Mp['alpha'], Mp['mu'], M3['alpha'], M3['mu']]
        
        ### df_individual[ID] ###
        # make a dictionary that has each individual's data
        df = pd.DataFrame({'time': M1['time']})
        for colname, df_waves in zip(colnames, dfs_waves):
            df[colname] = df_waves
        df_individual[ID] = df
        
        ### df_tsa ###
        # make a dataframe for time series analysis
        df_tsa += df_individual[ID].fillna(df_individual[ID].mean())
        
        ### df_individuals ###
        # make a dataframe for a bar plot
        for df_waves in dfs_waves:
            data = df_waves[167:].mean()
            mean_list.append(data)
        dfdict[ID] = mean_list
        mean_list = [] # make a list empty before a next repeatation
        df_individuals = pd.DataFrame.from_dict(dfdict)
        ### df_analysis ###
        # make a dataframe for analysis (e.g., bayesian estimation)
        df_individuals['obs1/prod/obs3'] = ['obs1', 'obs1', 'prod', 'prod', 'obs3', 'obs3'] * 2 

    df_individuals.insert(1, 'mean', df_individuals.loc[:, subjects[0]:subjects[-1]].mean(axis=1))
    df_tsa /= len(subjects) 
    
    return df_individual, df_tsa, df_individuals 
    
df_individual, df_tsa, df_individuals = import_data(sigma_threshold=2, ratio_to_baseline=True)

def plot_barplots() -> None:
    """ uses data from df_individuals """
    fs = 17
    fig, ax = plt.subplots(figsize=(10,8))
    ax = sns.barplot(data=df_individuals, x='conditions', y=df_individuals['mean'],
                     width=0.5, hue='obs1/prod/obs3')
    #ax.tick_params(labeltop=True, labelbottom=False, labelsize=12.5)
    plt.xticks(x=df_individuals['conditions'], rotation=45)
    #ax.xaxis.set_label_position('top') 
    ax.set_title('Relative change of power in each condition', fontsize=fs)
    ax.set_xlabel('Conditions', fontsize=fs)
    ax.set_ylabel('Relative change of power', fontsize=fs)
    ax.legend(fontsize=fs)
    plt.show()

plot_barplots()

def plot_dists(df: pd.DataFrame, bins: int) -> None:     
    """ df: pd.DataFrame
        bins: int """ 
    col1 = df.iloc[167:, 1:5]
    col2 = df.iloc[167:, 5:9]
    col3 = df.iloc[167:, 9:] 
    title = 'Distributions of data'
    fig, axs = plt.subplots(3, 4, figsize=(12,8), constrained_layout=True)
    fig.suptitle(title, fontsize=20)
    
    def plot_subplots(x, df, col_dot_columns):   
        for y, df_col, name in zip(range(4), df, col_dot_columns): 
            axs[x,y].hist(data=df, x=df_col, bins=bins, histtype='stepfilled')
            axs[x,y].set_title(name)
    plot_subplots(0, col1, col1.columns)
    plot_subplots(1, col2, col2.columns)
    plot_subplots(2, col3, col3.columns)            
    plt.show()
    
plot_dists(df_tsa, bins=50)

def plot_all_data(df: pd.DataFrame) -> None:
    """ df: pd.DataFrame """
    col1 = df.iloc[:, 1:5]
    col2 = df.iloc[:, 5:9]
    col3 = df.iloc[:, 9:]
    time = df['time']
    xticks = [-2, -1, 0, 1, 2, 3, 4]
    yticks = np.arange(0.6*1e-10, 2.1*1e-10, 0.2*1e-10)
    fig, axs = plt.subplots(3, 4, figsize=(15,10), constrained_layout=True)
    
    def plot_subplots(x: int, col: pd.DataFrame, col_columns: str) -> None:
        for y, power, title in zip(range(4), col, col.columns):
            axs[x,y].plot(time, power, data=col, alpha=0.7)
            axs[x,y].set_xticks(xticks)
            axs[x,y].set_yticks(yticks)
            axs[x,y].set_title(title) 
            
    plot_subplots(0, col1, col1.columns)
    plot_subplots(1, col2, col2.columns)
    plot_subplots(2, col3, col3.columns)  
    fs = 18
    fig.suptitle('Change of power in each condition', fontsize=fs)
    fig.supxlabel('Time', fontsize=fs)
    fig.supylabel('Power', fontsize=fs)
    plt.show()

plot_all_data(df=df_tsa)

def plot_data(df: pd.DataFrame, condition: str) -> None:
    """ df: pd.DataFrame
        condition: combination of (C/M)_(1/p/3)_(alpha/mu) """
    fig, ax = plt.subplots()
    time = df['time']
    power = df[condition]
    mean_power = power[167:].mean()*100
    baseline = df[:167]
    time_movie = df[167:]
    power_lt_zero = time_movie[time_movie < 0]
    power_gt_zero = time_movie[time_movie >= 0]
    ax.plot(baseline['time'], baseline[condition])
    ax.plot(time_movie['time'], power_lt_zero[condition])
    ax.plot(time_movie['time'], power_gt_zero[condition])
    kwargs = {'linestyle': '--', 'linewidth': 2.0, 'color': 'black', 'alpha': 0.3}
    plt.vlines(x=0, ymin=min(power), ymax=max(power), **kwargs)
    plt.title(condition, fontsize=14)
    plt.text(x=max(time)*0.3, y=max(power)*0.7, s=f'mean(0s-4s): {mean_power:.2f}%', fontsize=14)    
    plt.grid(True)
    plt.show()

for colname in colnames:
    plot_data(df_tsa, colname)
