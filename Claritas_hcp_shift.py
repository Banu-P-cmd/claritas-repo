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
unique_hcp_ids = data['hcp_id'].unique()
len(unique_hcp_ids)
#%%
baseline_period = '2021-01'
end_period = '2022-09'

# Ensure 'ship_date' is in PeriodDtype format for monthly granularity
data['ship_date'] = pd.to_datetime(data['ship_date']).dt.to_period('M')

# Correctly filter data for the period of interest with proper parentheses
data_period_of_interest = data[(data['ship_date'] >= baseline_period) & (data['ship_date'] <= end_period)]

# Filter data for the start and end periods
start_data = data[data['ship_date'] == baseline_period]
end_data = data[data['ship_date'] == end_period]
#%%
# Aggregate prescriptions by patient and brand for each period
start_prescriptions = start_data.groupby(['patient_id', 'brand']).size().unstack(fill_value=0)
end_prescriptions = end_data.groupby(['patient_id', 'brand']).size().unstack(fill_value=0)

# Ensure all expected brands are present in both DataFrames
for brand in ['A', 'B', 'C', 'D']:
    if brand not in start_prescriptions:
        start_prescriptions[brand] = 0
    if brand not in end_prescriptions:
        end_prescriptions[brand] = 0
#%%

# started with A,
def identify_hcp_shifts_from_brand_a(group):
    shifted_brands = []  # Tracks shifts of HCPs to other brands after Brand A
    prescribed_a = False

    for index, row in group.iterrows():
        # Check if Brand A is initially prescribed by the HCP
        if row['brand'] == 'A':
            prescribed_a = True
        elif prescribed_a and row['brand'] in ['B', 'C', 'D']:
            # Record every switch to B, C, or D after initial A prescription by the HCP
            shifted_brands.append(row['brand'])

    # Return the list of brands the HCP switched to after prescribing Brand A
    return shifted_brands

# Apply the function across the dataset grouped by HCP
all_hcp_switches = data.groupby('hcp_id').apply(identify_hcp_shifts_from_brand_a)

# Filter out empty lists to keep only HCPs who made switches
valid_hcp_switches = all_hcp_switches[all_hcp_switches.apply(len) > 0]

# Flatten the list of brands to which HCPs switched
all_shifted_brands = [brand for sublist in valid_hcp_switches for brand in sublist]

# Count the frequency of HCPs switching to each brand
hcp_switches_counter = Counter(all_shifted_brands)

# Identify the brand with the most switches by HCPs
most_switched_to_brand_by_hcps = hcp_switches_counter.most_common(1)[0] if hcp_switches_counter else ('None', 0)

print(f"Brand with the most switches by HCPs from A: {most_switched_to_brand_by_hcps[0]} with {most_switched_to_brand_by_hcps[1]} switches")

# The list of unique HCP IDs who made valid switches remains the same
print(f"Unique HCP IDs who switched from prescribing Brand A: {valid_hcp_switches.index.tolist()}")


#%%
len(valid_hcp_switches) # 1152 hcp's who never prescribed brand A again, after starting with it
#%%
most_switched_to_brand_by_hcps # brand with the most hcp switches
#%%
hcp_switches_counter
#%%
def identify_hcp_shifts_after_stopping_a(group):
    # Ensure the group is sorted by 'ship_date'
    group = group.sort_values(by='ship_date')

    # Identify if and when Brand A was last prescribed
    last_a_prescription = group[group['brand'] == 'A']['ship_date'].max()

    # If Brand A was never prescribed, return None
    if pd.isnull(last_a_prescription):
        return None

    # Filter for prescriptions after the last Brand A prescription
    shifted_after_a = group[group['ship_date'] > last_a_prescription]

    # If there are no prescriptions after the last Brand A prescription, return None
    if shifted_after_a.empty:
        return None

    # Get unique brands prescribed after the last Brand A prescription
    shifted_brands = shifted_after_a['brand'].unique().tolist()

    return shifted_brands

# Apply the function to each HCP's prescriptions
all_hcp_shifts_after_stopping_a = data.groupby('hcp_id').apply(identify_hcp_shifts_after_stopping_a)

# Drop None to keep only HCPs who shifted to prescribing other brands after stopping A
valid_hcp_shifts_after_stopping_a = all_hcp_shifts_after_stopping_a.dropna()

# Display the result # who stopped prescribing brand A completely after switching
print(f"HCPs who stopped prescribing Brand A and shifted to other brands: {valid_hcp_shifts_after_stopping_a}")


#%%
len(valid_hcp_shifts_after_stopping_a)
#%%
type(valid_hcp_shifts_after_stopping_a)
#%%
# Initialize a dictionary to count occurrences of each brand
brand_preference_count = {}

# Iterate through the dictionary to count each brand's occurrence
for hcp_id, brands in valid_hcp_shifts_after_stopping_a.items():
    for brand in brands:
        if brand in brand_preference_count:
            brand_preference_count[brand] += 1
        else:
            brand_preference_count[brand] = 1

# Now, brand_preference_count contains the count of each brand preferred by HCPs after stopping Brand A
print(brand_preference_count)
#%%
status_data = pd.read_csv('/Users/bn/Downloads/status.csv')
#data['ship_date'] = pd.to_datetime(data['ship_date'])
print(status_data.columns)
#%%
print("Overall Analysis:")
print(status_data[['status', 'status_detail', 'status_sub_detail']].describe(include='all'))


#%%
filtered_status_data = status_data[status_data['hcp_id'].isin(valid_hcp_shifts_after_stopping_a.index)]
print("\nFiltered Data Analysis:")
#print(filtered_status_data[['status', 'status_detail', 'status_sub_detail']].describe(include='all'))

#%%
# Comparison of the 'status' distribution between overall and filtered data
print("\nComparison of 'status' distribution:")
print("Overall 'status' distribution:")
print(status_data['status'].value_counts(normalize=True))

print("\nFiltered 'status' distribution:")
print(filtered_status_data['status'].value_counts(normalize=True))
#%%
print("\nComparison of 'status' detail distribution:")
print("Overall 'status' detail distribution:")
print(status_data['status_detail'].value_counts(normalize=True))

print("\nFiltered 'status' detail distribution:")
print(filtered_status_data['status_detail'].value_counts(normalize=True))
#%%
# Calculate the normalized value counts for 'status_detail' in both datasets
overall_distribution = status_data['status_detail'].value_counts(normalize=True)
filtered_distribution = filtered_status_data['status_detail'].value_counts(normalize=True)

# Calculate the differences in distributions
difference_distribution = overall_distribution.subtract(filtered_distribution, fill_value=0)

# Sort the differences by their absolute values in descending order and select the top 5
top_5_differences = difference_distribution.abs().sort_values(ascending=False).head(5)

print("Top 5 categories with the biggest difference in 'status_detail' distribution:")
print(top_5_differences)

#%%
# Calculate the normalized value counts for 'status_detail' in both datasets
overall_distribution = status_data['status_sub_detail'].value_counts(normalize=True)
filtered_distribution = filtered_status_data['status_sub_detail'].value_counts(normalize=True)

# Calculate the differences in distributions
difference_distribution = overall_distribution.subtract(filtered_distribution, fill_value=0)

# Sort the differences by their absolute values in descending order and select the top 5
top_5_differences = difference_distribution.abs().sort_values(ascending=False).head(5)

print("Top 5 categories with the biggest difference in 'status_detail' distribution:")
print(top_5_differences)

#%%

#This code will give me for each month's 'shipment scheduled' for a patient, the days difference for when was the
# latest referral that was gotten for that patient. And then you average it for each month. So for each month,
# you'll get an average number of days it took for the patient to get their drug scheduled.

status_data['status_dt'] = pd.to_datetime(status_data['status_dt'])

# Filter for 'referral' and 'shipment scheduled' statuses

# Ensure 'status_dt' is in datetime format
#sorted__status_data['status_dt'] = pd.to_datetime(sorted__status_data['status_dt'])

# Now continue with your existing code, as the 'status_dt' column is guaranteed to be in the correct format

relevant_statuses = status_data[status_data['status_detail'].isin(['referral', 'shipment scheduled'])]

# Sort data by patient and date
sorted__status_data = relevant_statuses.sort_values(by=['patient_id', 'status_dt'])

# Initialize a DataFrame to hold average times for each month
average_times = []

# Loop through each month in the range from Jan 2021 to Sep 2022
for month_end in pd.date_range(start='2021-01-01', end='2022-09-30', freq='M'):
    # The period we're looking at
    month_period = month_end.to_period('M')

    # Filter shipments for the current month
    shipments_this_month = sorted__status_data[(sorted__status_data['status_detail'] == 'shipment scheduled') &
                                               (sorted__status_data['status_dt'].dt.to_period('M') == month_period)]

    time_diffs = []  # To store time differences for this month

    # For each shipment this month, find the earliest referral
    for _, shipment in shipments_this_month.iterrows():
        patient_id = shipment['patient_id']
        shipment_date = shipment['status_dt']

        # Get all referrals for this patient before this shipment
        referrals = sorted__status_data[(sorted__status_data['patient_id'] == patient_id) &
                                (sorted__status_data['status_detail'] == 'referral') &
                                (sorted__status_data['status_dt'] < shipment_date)]

        if not referrals.empty:
            latest_referral_date = referrals['status_dt'].max()
            time_difference = (shipment_date - latest_referral_date).days
            time_diffs.append(time_difference)

    # Calculate the average time for the month if time_diffs is not empty
    if time_diffs:
        average_time = pd.Series(time_diffs).mean()
        average_times.append({'Year_Month': month_period, 'Average_Time_Days': average_time})

# Convert the list of averages to a DataFrame
average_times_df = pd.DataFrame(average_times)

print(average_times_df)
#%%
average_times_df['Year_Month'] = average_times_df['Year_Month'].astype(str)

plt.figure(figsize=(14, 7))
plt.plot(average_times_df['Year_Month'], average_times_df['Average_Time_Days'], marker='o', linestyle='-', color='blue')
plt.title('Average Time from Latest Referral to Shipment Scheduled')
plt.xlabel('Year-Month')
plt.ylabel('Average Time (Days)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
#%%
#This code will give me for each month's 'shipment scheduled' for a patient, the days difference for when was the
# latest referral that was gotten for that patient. And then you average it for each month. So for each month,
# you'll get an average number of days it took for the patient to get their drug scheduled.

#data_brand_a['status_dt'] = pd.to_datetime(data_brand_a['status_dt'])

# Filter for 'referral' and 'shipment scheduled' statuses
relevant_statuses = filtered_status_data[filtered_status_data['status_detail'].isin(['referral', 'shipment scheduled'])]

# Sort data by patient and date
sorted__filtered_status_data = relevant_statuses.sort_values(by=['patient_id', 'status_dt'])

# Initialize a DataFrame to hold average times for each month
average_times = []

# Loop through each month in the range from Jan 2021 to Sep 2022
for month_end in pd.date_range(start='2021-01-01', end='2022-09-30', freq='M'):
    # The period we're looking at
    month_period = month_end.to_period('M')

    # Filter shipments for the current month
    shipments_this_month = sorted__filtered_status_data[(sorted__filtered_status_data['status_detail'] == 'shipment scheduled') &
                                               (sorted__filtered_status_data['status_dt'].dt.to_period('M') == month_period)]

    time_diffs = []  # To store time differences for this month

    # For each shipment this month, find the earliest referral
    for _, shipment in shipments_this_month.iterrows():
        patient_id = shipment['patient_id']
        shipment_date = shipment['status_dt']

        # Get all referrals for this patient before this shipment
        referrals = sorted__status_data[(sorted__filtered_status_data['patient_id'] == patient_id) &
                                        (sorted__filtered_status_data['status_detail'] == 'referral') &
                                        (sorted__filtered_status_data['status_dt'] < shipment_date)]

        if not referrals.empty:
            latest_referral_date = referrals['status_dt'].max()
            time_difference = (shipment_date - latest_referral_date).days
            time_diffs.append(time_difference)

    # Calculate the average time for the month if time_diffs is not empty
    if time_diffs:
        average_time = pd.Series(time_diffs).mean()
        average_times.append({'Year_Month': month_period, 'Average_Time_Days': average_time})

# Convert the list of averages to a DataFrame
average_filtered_times_df = pd.DataFrame(average_times)

print(average_filtered_times_df)
#%%
average_filtered_times_df['Year_Month'] = average_times_df['Year_Month'].astype(str)

plt.figure(figsize=(14, 7))
plt.plot(average_times_df['Year_Month'], average_times_df['Average_Time_Days'], marker='o', linestyle='-', color='blue')
plt.title('Average Time from Latest Referral to Shipment Scheduled')
plt.xlabel('Year-Month')
plt.ylabel('Average Time (Days)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
#%%
