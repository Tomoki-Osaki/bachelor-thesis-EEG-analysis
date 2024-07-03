import pandas as pd
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
az.style.use("arviz-whitegrid")
az.style.use('arviz-grayscale')
SEED = 42

# import data and process
subjects = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', 
            '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', 
            '025', '026', '027', '028', '029', '030', '031', '032', '034', '036']

def import_data(sigma_threshold: int = None, 
                ratio_to_baseline: bool = False) -> pd.DataFrame:
    """ sigma_threshold: 2 or 3 to remove outliers
        ratio_to_baseline: if True, calculate ratio to baseline """
    # colnames = ['C1_alpha', 'C1_mu', 'Cp_alpha', 'Cp_mu', 'C3_alpha', 'C3_mu',
    #             'M1_alpha', 'M1_mu', 'Mp_alpha', 'Mp_mu', 'M3_alpha', 'M3_mu']
    colnames = ['Counting Pre alpha', 'Counting Pre mu', 'Counting Practice alpha', 
                'Counting Practice alph', 'Counting Post alpha', 'Counting Post mu',
                'Meaning Pre alpha', 'Meaning Pre mu', 'Meaning Practice alpha', 
                'Meaning Practice mu', 'Meaning Post alpha', 'Meaning Post mu']
    dfdict = {'conditions': colnames}
    mean_list = []
    df_individual = {}
    
    for ID in subjects:
        path = f'卒業論文/卒論 H201015 データファイル/EEG 実験データ/Mimic_{ID}/Mimic_{ID}_csv/'
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
        
        ### df_individuals ###
        # make a dataframe for a bar plot
        for df_waves in dfs_waves:
            data = df_waves[167:].mean()
            mean_list.append(data)
        dfdict[ID] = mean_list
        mean_list = [] # make a list empty before a next repeatation
        df_individuals = pd.DataFrame.from_dict(dfdict)
        df_individuals['obs1/prod/obs3'] = ['obs1', 'obs1', 'prod', 'prod', 'obs3', 'obs3'] * 2 

    df_individuals.insert(1, 'mean', df_individuals.loc[:, subjects[0]:subjects[-1]].mean(axis=1))
    
    return df_individuals
    
df_individuals = import_data(sigma_threshold=2, ratio_to_baseline=True)

def create_dfs() -> pd.DataFrame:
    df = df_individuals.iloc[[1, 5, 7, 11], :]
    df = df.drop(columns=['mean', 'obs1/prod/obs3'])
    df['conditions'] = 'counting', 'counting', 'meaning', 'meaning'
    df.insert(1, 'practice', ['obs1', 'obs3']*2)
    df = df.melt(id_vars=['conditions', 'practice'])
    df = df.drop(columns='variable')
    df = df.rename(columns={'value': 'ratio_to_baseline'})
    df_Cbase = df.replace(['counting', 'meaning', 'obs1', 'obs3'], [0, 1, 0, 1])
    df_Mbase = df.replace(['counting', 'meaning', 'obs1', 'obs3'], [1, 0, 0, 1])
    
    return df_Cbase, df_Mbase

df_Cbase, df_Mbase = create_dfs()

y = df_Cbase['ratio_to_baseline']
x1 = df_Cbase['conditions']
x2 = df_Cbase['practice']
x3 = x1 * x2 

# bayesian regression using pymc package
with pm.Model() as model:
    # define priors
    lower, upper = -1, 1
    sigma = pm.Uniform('sigma', lower=0.01, upper=1)
    Intercept = pm.Uniform('Intercept', lower=lower, upper=upper)
    conditions = pm.Uniform('conditions', lower=lower, upper=upper)
    practice = pm.Uniform('practice', lower=lower, upper=upper)
    interaction = pm.Uniform('interaction', lower=lower, upper=upper)
    
    # formula
    formula = Intercept + conditions*x1 + practice*x2 + interaction*x3
    
    # define likelihood
    lh = pm.Normal('y', mu=formula, sigma=sigma, observed=y)
    
    # inference
    idata = pm.sample(draws=2000, tune=1000, chains=4, random_seed=SEED)

kwargs = {'hdi_prob': 0.95, 'rope': (-0.01, 0.01), 'textsize': 13}

az.plot_trace(idata, combined=False, legend=True)
az.plot_forest(idata, **kwargs)
az.plot_posterior(idata, figsize=(15, 8), ref_val=0, **kwargs, color='black')

result = az.summary(idata, hdi_prob=kwargs['hdi_prob'], round_to=3)
result.insert(0, 'Parameters', result.index)
result.drop(columns=['ess_bulk', 'ess_tail'], inplace=True)
names_to_change = {'mean': 'Mean',
                   'sd': 'SD',
                   'hdi_2.5%': 'HDI 2.5%',
                   'hdi_97.5%': 'HDI 97.5%',
                   'mcse_mean': 'MCSE mean',
                   'mcse_sd': 'MCSE SD',
                   'r_hat': 'R-hat'}
result.rename(columns=names_to_change, inplace=True)

import plotly.figure_factory as ff
fig = ff.create_table(result, index=False)
fig.update_layout(autosize=False, width=1000)
fig.write_image("summary_table.png", )

# parameters of t-distribution
# nu = pm.Exponential("nu", 1 / 29.0) + 1
# lambda_t = 1 / sigma**2

# lh = pm.StudentT('y', mu=formula, nu=nu, lam=lambda_t, observed=y)

# # bayesian regression using bambi package
# import bambi as bmb
# fml = 'ratio_to_baseline ~ conditions * practice'

# modelC = bmb.Model(formula=fml, data=df_Cbase, categorical=['conditions', 'practice'])
# resultsC = modelC.fit(draws=2000, chains=4, random_seed=SEED)
# az.plot_trace(resultsC, figsize=(12, 8))
# az.plot_posterior(resultsC, ref_val=0, **kwargs)
# summaryC = az.summary(resultsC, hdi_prob=kwargs['hdi_prob'])

# modelM = bmb.Model(formula=fml, data=df_Mbase, categorical=['conditions', 'practice'])
# resultsM = modelM.fit(draws=2000, chains=4, random_seed=SEED)
# az.plot_trace(resultsM, figsize=(12, 8))
# az.plot_posterior(resultsM, ref_val=0, **kwargs)
# summaryM = az.summary(resultsM, hdi_prob=kwargs['hdi_prob'])
