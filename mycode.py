import pandas as pd
import os

#create a sample Dataframe with column names
data = {"Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "City": ["New York", "Los Angeles", "Chicago"]
        }

df = pd.DataFrame(data)

# adding a new row to the DF for v2
new_row_loc = {"Name": "v2", "Age": 20, "City": "City1"}
df.loc[len(df.index)] = new_row_loc

# adding a new row to the DF for v3
new_row_loc2 = {"Name": "v3", "Age": 30, "City": "City1"}
df.loc[len(df.index)] = new_row_loc2

# Ensure the "data" directory exists at the root level
data_dir = "data"
os.makedirs(data_dir, exist_ok = True)

# define the file path
file_path = os.path.join(data_dir, "sample_data.csv")

# save the Dataframe to a csv file, including column names
df.to_csv(file_path, index=False)

print(f"CSV file saved to {file_path}")

  

