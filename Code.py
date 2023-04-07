import statsmodels.api as sm
import pandas as pd
from scipy import stats
import statsmodels.stats.multicomp as mc
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Generate random data
data = {'Treatment': ['100 kg', '200 kg', '300 kg'] * 3,
        'Replicate': [1, 2, 3] * 3,
        'Yield': [50, 55, 53, 45, 50, 48, 65, 62, 66],
        'Weight': [2.1, 2.3, 2.4, 1.8, 1.9, 2.0, 2.8, 2.9, 2.6]}
df = pd.DataFrame(data)

# ANOVA for Yield
model_yield = smf.ols('Yield ~ C(Treatment) + C(Replicate)', data=df).fit()
aov_table_yield = sm.stats.anova_lm(model_yield, typ=2)

# DMRT for Yield
comp_yield = mc.MultiComparison(df['Yield'], df['Treatment'])
posthoc_yield = comp_yield.tukeyhsd()
posthoc_summary_yield = posthoc_yield.summary()

# Plot mean values with error bars for Yield
means_yield = df.groupby(['Treatment']).mean()
stds_yield = df.groupby(['Treatment']).std()
means_yield.plot(kind='bar', y='Yield', yerr=stds_yield, capsize=4)
plt.xticks(rotation=360)
plt.legend(title='Treatment', loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('Yield (kg/ha)')
plt.show()

# ANOVA for Weight
model_weight = smf.ols('Weight ~ C(Treatment) + C(Replicate)', data=df).fit()
aov_table_weight = sm.stats.anova_lm(model_weight, typ=2)

# DMRT for Weight
comp_weight = mc.MultiComparison(df['Weight'], df['Treatment'])
posthoc_weight = comp_weight.tukeyhsd()
posthoc_summary_weight = posthoc_weight.summary()

# Plot mean values with error bars for Weight
means_weight = df.groupby(['Treatment']).mean()
stds_weight = df.groupby(['Treatment']).std()
means_weight.plot(kind='bar', y='Weight', yerr=stds_weight, capsize=4)
plt.xticks(rotation=360)
plt.legend(title='Treatment', loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('Weight (kg)')
plt.show()

print(aov_table_yield)
print(posthoc_summary_yield)

print(aov_table_weight)
print(posthoc_summary_weight)
