import pandas as pd

subjects = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', 
            '011', '012', '013', '014', '015', '016', '017', '018', '019', '020',
            '021', '022', '025', '026', '027', '028', '029', '030', '031', '032',
            '034', '036']

colnames = ['kazu_obs1_alpha', 'kazu_obs1_mu', 'kazu_prod_alpha', 'kazu_prod_mu', 'kazu_obs2_alpha', 'kazu_obs2_mu',
            'imi_obs1_alpha', 'imi_obs1_mu', 'imi_prod_alpha', 'imi_prod_mu', 'imi_obs2_alpha', 'imi_obs2_mu']

def import_data(sigma_threshold: int = None, 
                ratio_to_baseline: bool = False) -> pd.DataFrame:
    """ sigma_threshold (number): Specify what sigma threshold to be used (usually 2 or 3) to 
        remove outliers when calculating individual's ratio to baseline values. 
        If None (by default), does not remove outliers. """
        
    power_list, baseline_list = [],[]
    df_analysis = pd.DataFrame({'conditions': colnames, 'ID': 0, 'power': 0, 'baseline': 0})
    
    for ID in subjects:
        path = f'C:/Users/ootmo/OneDrive/Documents/卒業論文/卒論 H201015 データファイル/EEG 実験データ/Mimic_{ID}/Mimic_{ID}_csv/'
        kazu_obs1 = pd.read_csv(path + f'Mimic_{ID}_kazu_obs1.csv').groupby(['time'], as_index=False).mean()
        kazu_prod = pd.read_csv(path + f'Mimic_{ID}_kazu_prod.csv').groupby(['time'], as_index=False).mean()
        kazu_obs2 = pd.read_csv(path + f'Mimic_{ID}_kazu_obs2.csv').groupby(['time'], as_index=False).mean()
        imi_obs1 = pd.read_csv(path + f'Mimic_{ID}_imi_obs1.csv').groupby(['time'], as_index=False).mean()
        imi_prod = pd.read_csv(path + f'Mimic_{ID}_imi_prod.csv').groupby(['time'], as_index=False).mean()
        imi_obs2 = pd.read_csv(path + f'Mimic_{ID}_imi_obs2.csv').groupby(['time'], as_index=False).mean()    
        dfs = [kazu_obs1, kazu_prod, kazu_obs2, imi_obs1, imi_prod, imi_obs2]
    
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
        dfs_waves = [
            kazu_obs1['alpha'], kazu_obs1['mu'], kazu_prod['alpha'], kazu_prod['mu'], kazu_obs2['alpha'], kazu_obs2['mu'], 
            imi_obs1['alpha'], imi_obs1['mu'], imi_prod['alpha'], imi_prod['mu'], imi_obs2['alpha'], imi_obs2['mu']]
        
        ### df_individuals ###
        # make a dataframe for a bar plot
        for df_wave in dfs_waves:
            power = df_wave[167:].mean()
            power_list.append(power)
            baseline = df_wave[:167].mean()
            baseline_list.append(baseline)
        indv = pd.DataFrame({'conditions': colnames, 'ID': 0, 'power': 0, 'baseline': 0})
        indv['ID'] = ID
        indv['power'] = power_list
        indv['baseline'] = baseline_list
        power_list, baseline_list = [],[]
        df_analysis = pd.concat([df_analysis, indv], join='inner', ignore_index=True)
        
    df_analysis.drop(df_analysis.index[:12], inplace=True)
    df_analysis.index = df_analysis['conditions'] 
    df_analysis.drop(columns='conditions', inplace=True)
    
    return df_analysis 