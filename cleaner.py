import pandas as pd

data = pd.read_csv('stress_messages.csv')
data = data.drop_duplicates()