import pandas as pd
import os
import sys
from settings import PROJECT_ROOT
filename = 'file_test.csv'
data_path = os.path.join(PROJECT_ROOT, 'data', filename)
df = pd.read_csv(data_path, index_col=0)
