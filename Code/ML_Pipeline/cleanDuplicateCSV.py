import pandas as pd

df = pd.read_csv("experiment_results.csv", sep=None, engine='python')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp', ascending=True)
df = df.drop_duplicates(subset=['Experiment', 'Model', 'Region', 'Target'], keep='last')
df = df.sort_values(['Experiment', 'Model'])
df.to_csv("experiment_results_clean.csv", index=False)
print(f"Cleaned: {len(df)} unique rows remaining")