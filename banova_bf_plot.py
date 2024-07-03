import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = "Pythonfiles_bachelor_thesis/BF_df.csv"
BF_df = pd.read_csv(path)
BF_df.drop(index=2, inplace=True)
BF_df.drop(columns=['Unnamed: 0'], inplace=True)
BF_df['models'] = ['practice', 'task', 'practice + task + interaction']

evidences = []
for bf in BF_df['BF01']:
    if bf > 30: evidences.append('Very strong evidence (30-100)')
    elif bf < 30 and bf > 10: evidences.append('Strong evidence (10-30)')
    elif bf < 10 and bf > 3: evidences.append('Moderate evidence (3-10)')
    else: evidences.append('Anecdotal evidence (1-3)')
BF_df['Strength of evidence'] = evidences

plt.rcParams["font.size"] = 15
fig, ax = plt.subplots(1, 1, figsize=[10, 6])
ax = sns.barplot(data=BF_df, x='models', y='BF01', width=0.5)
ax.set_title('BF01 in each model') 
ax.set_xlabel('\nModel', fontdict={'size': 17})
ax.set_ylabel('BF01 (Log Scale)', fontdict={'size': 17})
ax.set_yscale('log')
for ycoord in [3, 10, 30]:
    ax.axhline(ycoord, linewidth=1.8, linestyle='dotted', color='grey')
xpos = 2.65
ax.text(x=xpos, y=33, s='Very strong evidence (30-100)')
ax.text(x=xpos, y=17, s='Strong evidence (10-30)')
ax.text(x=xpos, y=5.8, s='Moderate evidence (3-10)')
ax.text(x=xpos, y=2.1, s='Anecdotal evidence (1-3)')
xpos2 = 2.52
ax.text(x=xpos2, y=28, s='30')
ax.text(x=xpos2, y=9.5, s='10')
ax.text(x=xpos2, y=2.9, s='3')
ax.text(x=xpos2, y=1.6, s='1')
plt.show()

"""
BF01            Evidence category (for H0)
--------------------------------------------
1                      No Evidence
1 - 3               Anecdotal evidence
3 - 10               Moderate evidence
10 - 30                Strong evidence
30 - 100             Very strong evidence
>100                  Extreme evidence
"""

