import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport

#importing the data
df = sns.load_dataset('penguins')

#descriptive statistics
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("Profiling_report.html")