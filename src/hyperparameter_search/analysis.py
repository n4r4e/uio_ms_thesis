#%%
###########################################################################
# Calculate mean and std of similarity scores for each model
###########################################################################
import pandas as pd

df = pd.read_csv('all_with_similarity_scores.csv')

# Calculate mean and std of similarity scores for each model
model_avg_stats = df.groupby('model')['similarity_score'].agg(['mean', 'std']).sort_values(by='mean', ascending=False)
print(model_avg_stats)

model_avg_stats.to_csv('hyperparam_similarity_scores.csv')

#%%
###########################################################################
# Visualization of mean similarity scores for each model
###########################################################################
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Create bar graph of mean values for each model
model_avg_stats['mean'].plot(kind='bar', yerr=model_avg_stats['std'], capsize=4)

plt.ylabel('Average Similarity Score')
plt.xlabel('Model')

plt.tight_layout()
plt.savefig('hyperparam_similarity_score.pdf')
plt.show()

#%%
###########################################################################
# Extract top 5 hyperparameter configurations with the highest mean similarity score for each model
###########################################################################
import pandas as pd

df = pd.read_csv('all_with_similarity_scores.csv')

# Group by hyperparameter configuration and calculate mean and std of similarity scores
grouped = df.groupby(['model', 'do_sample', 'num_beams', 'temperature', 'repetition_penalty'])

# Calculate mean and std
stats = grouped['similarity_score'].agg(['mean', 'std']).reset_index()

# Extract top 5 hyperparameter configurations with the highest mean similarity score for each model
top_configs = stats.groupby('model').apply(lambda x: x.nlargest(5, 'mean')).reset_index(drop=True)

top_configs.to_csv('top_hyperparameter_configurations.csv', index=False)
#%%
###########################################################################
# Visualization of Top 5 Hyperparameter Configurations 
###########################################################################
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('top_hyperparameter_configurations.csv')

# Get model names
models = df['model'].unique()

# Set subplots 
fig, axs = plt.subplots(6, 4, figsize=(20, 25), sharey=True)  # Share y-axis

# Draw subplots for each model
for i, model in enumerate(models):
    ax = axs[i // 4, i % 4]  # Calculate row and column indices
    model_data = df[df['model'] == model]
    x = range(len(model_data))
    ax.bar(x, model_data['mean'], yerr=model_data['std'], capsize=5, alpha=0.7)
    ax.set_title(model, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels('T '+ model_data['temperature'].astype(str) + ', R ' + model_data['repetition_penalty'].astype(str), rotation=45, fontsize=14)

plt.tight_layout()
plt.savefig('hyperparam_top5_configs.pdf')
plt.show()