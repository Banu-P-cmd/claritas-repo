import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

#%%
pd.set_option('display.max_rows', None)
#%%
data = pd.read_csv('/Users/bn/Downloads/ship.csv')
#data['ship_date'] = pd.to_datetime(data['ship_date'])
print(data.columns)
#%%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming data for all brands is in a DataFrame called 'data_all_brands'
# and it includes a 'brand' column distinguishing between brands A, B, C, and D.

# Ensure 'ship_date' is a datetime column
data['ship_date'] = pd.to_datetime(data['ship_date'])

# Define the brands to analyze
brands = ['A', 'B', 'C', 'D']

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

# Loop through each brand to aggregate and plot data
for brand in brands:
    # Filter data for the current brand
    data_brand = data[data['brand'] == brand]

    # Aggregate data
    monthly_out_of_pocket_cost = data_brand.groupby(data_brand['ship_date'].dt.to_period('M'))['total_out_of_pocket_cost'].mean()
    monthly_plan_paid_amount = data_brand.groupby(data_brand['ship_date'].dt.to_period('M'))['total_plan_paid_amount'].mean()

    # Convert the PeriodIndex back to datetime for plotting
    monthly_out_of_pocket_cost.index = monthly_out_of_pocket_cost.index.to_timestamp()
    monthly_plan_paid_amount.index = monthly_plan_paid_amount.index.to_timestamp()

    # Plotting
    axs[0].plot(monthly_out_of_pocket_cost.index, monthly_out_of_pocket_cost, label=f"Brand {brand}")
    axs[1].plot(monthly_plan_paid_amount.index, monthly_plan_paid_amount, label=f"Brand {brand}")

# Setting legends and titles
axs[0].legend()
axs[0].set_title('Average Out-of-Pocket Cost by Month')
axs[0].set_ylabel('Average Cost ($)')

axs[1].legend()
axs[1].set_title('Average Plan Paid Amount by Month')
axs[1].set_ylabel('Average Amount ($)')
axs[1].set_xlabel('Month')

plt.tight_layout()
plt.show()


#%%
