import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

#%%
pd.set_option('display.max_rows', None)
#%%
data = pd.read_csv('/Users/bn/Downloads/ship.csv')
#data['ship_date'] = pd.to_datetime(data['ship_date'])


#%%

#%%


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
#data_period_of_interest = data[(data['ship_date'] >= baseline_period) & (data['ship_date'].dt.to_period('M') <= end_period)]
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
brand_a_prescriptions = data_period_of_interest[data_period_of_interest['brand'] == 'A']

#%%
# Group by patient and brand, then sort by ship_date to track prescription history
sorted_prescriptions = data_period_of_interest.sort_values(by=['patient_id', 'ship_date'])

# Find patients who were prescribed Brand A and then any other brand
patients_started_with_a = sorted_prescriptions[sorted_prescriptions['patient_id'].isin(brand_a_prescriptions['patient_id'])]
#%%

#%%
# Filter data for the period of interest and for Brand A only
brand_a_data_period = data[(data['ship_date'] >= baseline_period) & (data['ship_date'] <= end_period) & (data['brand'] == 'A')]

# Find the first prescription date for each patient within the filtered data
first_prescription_dates = brand_a_data_period.sort_values(by=['patient_id', 'ship_date']).drop_duplicates('patient_id', keep='first')

# Extract the year and month from 'ship_date' for grouping
first_prescription_dates['year_month'] = first_prescription_dates['ship_date'].dt.strftime('%Y-%m')

# Group by the extracted year and month, then count unique patients
new_patients_per_month = first_prescription_dates.groupby('year_month')['patient_id'].nunique()

print("Number of new patients for Brand A each month within the period:")
print(new_patients_per_month)


#%%

#%%
# Filter data for the period of interest and for Brands A, B, C, D
data_period_of_interest = data[(data['ship_date'] >= baseline_period) & (data['ship_date'] <= end_period) & (data['brand'].isin(['A', 'B', 'C', 'D']))]

# Find the first prescription date for each patient within the filtered data for each brand
first_prescription_dates = data_period_of_interest.sort_values(by=['patient_id', 'ship_date']).drop_duplicates(subset=['patient_id', 'brand'], keep='first')

# Extract the year and month from 'ship_date' for grouping
first_prescription_dates['year_month'] = first_prescription_dates['ship_date'].dt.strftime('%Y-%m')

# Group by brand and the extracted year and month, then count unique patients
new_patients_per_month_per_brand = first_prescription_dates.groupby(['brand', 'year_month'])['patient_id'].nunique()

print("Number of new patients for each brand each month within the period:")
print(new_patients_per_month_per_brand)
#%%
# Assuming new_patients_per_month_per_brand is your DataFrame or Series with the counts
# If it's not already in a suitable format (e.g., DataFrame with columns for 'brand', 'year_month', and 'count'),
# you might need to reset the index if your data is in a MultiIndex Series.
if isinstance(new_patients_per_month_per_brand, pd.Series):
    new_patients_per_month_per_brand = new_patients_per_month_per_brand.reset_index(name='count')

# Pivot the data for plotting. This transforms the data so that:
# - each brand is a column,
# - each row represents a month,
# - cell values represent the count of new patients.
pivot_data = new_patients_per_month_per_brand.pivot(index='year_month', columns='brand', values='count')

# Plotting
pivot_data.plot(kind='bar', figsize=(14, 7), width=0.8)
plt.title('Number of New Patients per Month per Brand')
plt.xlabel('Year and Month')
plt.ylabel('Number of New Patients')
plt.xticks(rotation=45)
plt.legend(title='Brand')
plt.tight_layout()  # Adjust the layout to make room for the rotated x-axis labels

plt.show()
#%%
# Summarize prescriptions by patient across the entire period
prescription_summary = data_period_of_interest.groupby(['patient_id', 'brand']).size().unstack(fill_value=0)

# Identify patients with prescriptions for Brand A and any competitors
patients_with_brand_a = prescription_summary[prescription_summary['A'] > 0]
patients_with_competitors = prescription_summary[(prescription_summary[['B', 'C', 'D']].sum(axis=1)) > 0]

# Find unique patient_ids for those who have been prescribed both Brand A and any of the competitors
common_patient_ids = set(patients_with_brand_a.index) & set(patients_with_competitors.index)

# Convert set to list if needed for further processing
unique_patient_ids_with_both = list(common_patient_ids)
#%%
display(unique_patient_ids_with_both)
len(unique_patient_ids_with_both)
#%%

#%%
# # Identify patients who had prescriptions for Brand A initially but then shifted
# shifted_patients = start_prescriptions[start_prescriptions['A'] > 0].index.intersection(end_prescriptions[end_prescriptions['A'] == 0].index)
#
# # For these patients, check if they have increased prescriptions for other brands
# shifted_prescriptions = end_prescriptions.loc[shifted_patients]
# shifted_prescriptions = shifted_prescriptions[(shifted_prescriptions[['B', 'C', 'D']] > start_prescriptions.loc[shifted_patients][['B', 'C', 'D']]).any(axis=1)]
#
# # Count how many patients increased their prescriptions for each brand
# brand_shift_counts = (shifted_prescriptions[['B', 'C', 'D']] > start_prescriptions.loc[shifted_prescriptions.index][['B', 'C', 'D']]).sum()
#
# # Determine the brand with the most significant increase in patient count
#
# most_increased_brand = brand_shift_counts.idxmax()
# most_increased_count = brand_shift_counts.max()
#
# print(f"Brand with the most increased patient count due to loyalty shifts: {most_increased_brand} with {most_increased_count} patients")

#%%
#display(brand_shift_counts)
#%%
# Find common patients in both periods
# common_patients = start_prescriptions.index.intersection(end_prescriptions.index)
#
# # Safely select only those common patients for comparison
# start_prescriptions_common = start_prescriptions.loc[common_patients]
# end_prescriptions_common = end_prescriptions.loc[common_patients]
#
# # Proceed with your logic for identifying shifted prescriptions
# # This ensures you're only working with patients present in both periods, avoiding KeyError
# shifted_prescriptions_increase = end_prescriptions_common[
#     (end_prescriptions_common[['B', 'C', 'D']] > start_prescriptions_common[['B', 'C', 'D']]).any(axis=1)
#]
#%%
#display(shifted_prescriptions_increase)
#%%
# # Aggregate prescriptions by patient, brand, and month
# monthly_prescriptions = data_period_of_interest.groupby(['patient_id', 'brand', data['ship_date'].dt.strftime('%Y-%m')]).size().unstack(fill_value=0).reset_index()
#
# # Pivot the data to have one row per patient and columns for each brand and month
# pivot_df = monthly_prescriptions.pivot(index='patient_id', columns='brand', values=data['ship_date'].dt.strftime('%Y-%m'))
#
# # Identify patients who started with Brand A
# patients_started_with_a = pivot_df[pivot_df.groupby('patient_id')['A'].cumsum() > 0]
#
# # Find those who later had prescriptions for Brands B, C, or D
# shifted_patients = patients_started_with_a[patients_started_with_a[['B', 'C', 'D']].cumsum(axis=1).any(axis=1)]
#
# # Extract patient IDs who also have prescriptions for other brands
# shifted_patient_ids = shifted_patients.index.unique()
#
# # Display or process the list of patient IDs who shifted from Brand A to other brands
# print("Patients who shifted from Brand A to other brands:", shifted_patient_ids.tolist())
# #%%
# unique_patient_ids = data['patient_id'].unique()
# unique_patient_ids
#%%
# Identify all instances where Brand A was prescribed

#%%
# For each patient, identify if they have prescriptions for other brands after Brand A
def check_shift_to_other_brands(group):
    # Track if Brand A was prescribed first
    prescribed_a = False
    for index, row in group.iterrows():
        if row['brand'] == 'A':
            prescribed_a = True
        elif prescribed_a and row['brand'] in ['B', 'C', 'D']:
            return True
    return False

shifted_patients = patients_started_with_a.groupby('patient_id').apply(check_shift_to_other_brands)

# List of patient IDs who shifted from Brand A to other brands
shifted_patient_ids = shifted_patients[shifted_patients].index.tolist()

#%%
display(shifted_patient_ids)
len(shifted_patient_ids)
#%%
# from collections import Counter
#
# # Modified function to track brands to which patients shifted
# def check_and_return_shifts(group):
#     shifts = []  # To track shifts to other brands after Brand A
#     prescribed_a = False
#     for index, row in group.iterrows():
#         if row['brand'] == 'A':
#             prescribed_a = True
#         elif prescribed_a and row['brand'] in ['B', 'C', 'D']:
#             shifts.append(row['brand'])
#     return shifts
#
# # Apply the modified function to track shifts
# brand_shifts = patients_started_with_a.groupby('patient_id').apply(check_and_return_shifts)
#
# # Flatten the list of lists to a single list of brands
# all_shifts = [brand for sublist in brand_shifts for brand in sublist]
#
# # Count the frequency of each brand in the shifts
# brand_counts = Counter(all_shifts)
#
# # Identify which brand gained the most patients
# most_gained_brand = brand_counts.most_common(1)[0] if brand_counts else ('None', 0)
#
# print(f"Brand with the most patients shifted from A: {most_gained_brand[0]} with {most_gained_brand[1]} patients")

#%%

#%%

#%%
# def check_and_return_final_shifts(group):
#     final_shifts = {}
#     prescribed_a = False
#     stopped_a = False
#     for index, row in group.iterrows():
#         if row['brand'] == 'A':
#             if stopped_a:
#                 stopped_a = False  # Reset if A is prescribed again after a shift
#             prescribed_a = True
#         elif prescribed_a and row['brand'] in ['B', 'C', 'D']:
#             final_shifts[row['patient_id']] = row['brand']
#             stopped_a = True  # Mark that a shift occurred without returning to A
#     return final_shifts if stopped_a else {}
#
# # Apply the function to track final shifts without returning to Brand A
# final_shifts = sorted_prescriptions.groupby('patient_id').apply(check_and_return_final_shifts)
#
# # Extract patient IDs who made a final shift
# shifted_patient_ids = [pid for shifts in final_shifts for pid in shifts.keys()]
#
# # Use patient IDs to filter the original data for full records
# final_shifted_patients_df = data[data['patient_id'].isin(shifted_patient_ids)]
#
# # Count the frequency of shifts to each brand
# shifts_counter = Counter([shift for shifts in final_shifts for shift in shifts.values()])
#
# # Identify the brand with the most final shifts
# most_gained_brand_final = shifts_counter.most_common(1)[0] if shifts_counter else ('None', 0)
#
# print(f"Brand with the most patients who completely stopped Brand A: {most_gained_brand_final[0]} with {most_gained_brand_final[1]} patients")

#%%
# Updated function to track final shifts without returning to Brand A
# def check_and_return_final_shifts(group):
#     final_shifts = {}
#     prescribed_a = False
#     stopped_a = False
#     for index, row in group.iterrows():
#         if row['brand'] == 'A':
#             if stopped_a:
#                 stopped_a = False  # Reset if A is prescribed again after a shift
#             prescribed_a = True
#         elif prescribed_a and row['brand'] in ['B', 'C', 'D']:
#             final_shifts[row['patient_id']] = row['brand']
#             stopped_a = True  # Mark that a shift occurred without returning to A
#     return final_shifts if stopped_a else {}
#
# # Apply the function to track final shifts without returning to Brand A
# final_shifts = sorted_prescriptions.groupby('patient_id').apply(check_and_return_final_shifts)
#
# # Extract patient IDs who made a final shift
# shifted_patient_ids = [pid for shifts in final_shifts for pid in shifts.keys()]
#
# # Use patient IDs to filter the original data for full records
# final_shifted_patients_df = data[data['patient_id'].isin(shifted_patient_ids)]
#
# # Count the frequency of shifts to each brand
# shifts_counter = Counter([shift for shifts in final_shifts for shift in shifts.values()])
#
# # Identify the brand with the most final shifts
# most_gained_brand_final = shifts_counter.most_common(1)[0] if shifts_counter else ('None', 0)
#
# print(f"Brand with the most patients who completely stopped Brand A: {most_gained_brand_final[0]} with {most_gained_brand_final[1]} patients")

#%%
#display(final_shifted_patients_df)
#%%
#print(shifts_counter)
#%%
#len(shifted_patient_ids)
#%%
# counts only the final event
def check_and_return_final_shifts(group):
    final_shift = None  # Tracks the final shift to another brand, if any
    prescribed_a = False
    stopped_a = False
    for index, row in group.iterrows():
        if row['brand'] == 'A':
            if stopped_a:
                stopped_a = False  # Reset if A is prescribed again after a shift
            prescribed_a = True
        elif prescribed_a and row['brand'] in ['B', 'C', 'D']:
            final_shift = row['brand']
            stopped_a = True  # Mark that a shift occurred without returning to A
    return final_shift if stopped_a else None

# Apply the function and gather unique patient IDs who made a final shift
final_shifts = sorted_prescriptions.groupby('patient_id').apply(check_and_return_final_shifts)

# Filter out None values to keep only patients who made a final shift
final_shifts = final_shifts[final_shifts.notna()]

# Now final_shifts contains the brand each patient shifted to, without returning to Brand A
shifted_patient_ids = final_shifts.index.unique().tolist()

# Use patient IDs to filter the original data for full records
final_shifted_patients_df = data[data['patient_id'].isin(shifted_patient_ids)]

# Count the frequency of shifts to each brand
shifts_counter = Counter(final_shifts.values)

# Identify the brand with the most final shifts
most_gained_brand_final = shifts_counter.most_common(1)[0] if shifts_counter else ('None', 0)

print(f"Brand with the most patients who completely stopped Brand A: {most_gained_brand_final[0]} with {most_gained_brand_final[1]} patients")
print(f"Unique patient IDs who shifted: {shifted_patient_ids}")

#%%
display(shifted_patient_ids)
#%%
display(final_shifted_patients_df)

#%%
unique_patient_ids = final_shifted_patients_df['patient_id'].unique()
unique_patient_ids
#%%
len(shifted_patient_ids)
#%%
shifts_counter
#%%
# Function to identify any shifts to other brands after initially prescribing Brand A
# doesn't matter if they stopped/didn't stop taking Brand A
def identify_shifts(group):
    prescribed_a = False
    shifts = []  # To track shifts to other brands after Brand A
    for index, row in group.iterrows():
        if row['brand'] == 'A':
            prescribed_a = True
        elif prescribed_a and row['brand'] in ['B', 'C', 'D']:
            shifts.append(row['brand'])
    return shifts

# Apply the function and gather patient IDs with their shifts
shifts = sorted_prescriptions.groupby('patient_id').apply(identify_shifts)

# Keep only patients who made shifts
shifted_patients = shifts[shifts.apply(len) > 0]

# Flatten the list of brands to which patients shifted
all_shifts = [brand for sublist in shifted_patients for brand in sublist]

# Count the frequency of shifts to each brand
shifts_counter = Counter(all_shifts)

# Identify the brand with the most shifts
most_gained_brand = shifts_counter.most_common(1)[0] if shifts_counter else ('None', 0)

print(f"Brand with the most shifts from A: {most_gained_brand[0]} with {most_gained_brand[1]} shifts")

# Extract unique patient IDs who made shifts
shifted_patient_ids = shifted_patients.index.unique().tolist()

print(f"Unique patient IDs who shifted: {shifted_patient_ids}")
#%%
shifts_counter
#%%
# stopped taking brand A after the switch, and counts each respective switch as a single event, not just counts the final shift
def identify_switches_and_stop_brand_a(group):
    shifts = []  # Tracks all switches to other brands after Brand A
    prescribed_a = False
    ever_returned_to_a = False  # Flag to check if returned to A after initial switch

    for index, row in group.iterrows():
        # Mark when Brand A is initially prescribed
        if row['brand'] == 'A':
            if not prescribed_a:
                prescribed_a = True
            elif prescribed_a and len(shifts) > 0:
                # If returned to A after initial switch, invalidate all switches
                ever_returned_to_a = True
                break
        elif prescribed_a and row['brand'] in ['B', 'C', 'D']:
            # Record every switch to B, C, or D after initial A prescription
            shifts.append(row['brand'])

    # Only return shifts if there was no return to Brand A after the initial switch
    return shifts if not ever_returned_to_a else []

# Apply the function to each patient's group
all_switches = sorted_prescriptions.groupby('patient_id').apply(identify_switches_and_stop_brand_a)

# Keep only patients who made switches and never returned to Brand A
valid_switches = all_switches[all_switches.apply(len) > 0]

# Flatten the list of brands to which patients switched
all_switches_flattened = [brand for sublist in valid_switches for brand in sublist]

# Count the frequency of switches to each brand
switches_counter = Counter(all_switches_flattened)

# Identify the brand with the most switches
most_switched_to_brand = switches_counter.most_common(1)[0] if switches_counter else ('None', 0)

print(f"Brand with the most switches from A (ensuring no return to A): {most_switched_to_brand[0]} with {most_switched_to_brand[1]} switches")

# Extract unique patient IDs who made valid switches
valid_switched_patient_ids = valid_switches.index.unique().tolist()

print(f"Unique patient IDs who switched and never returned to A: {valid_switched_patient_ids}")


#%%
print(data.columns)
#%%
len(valid_switched_patient_ids)
#%%
filtered_data = data[data['patient_id'].isin(valid_switched_patient_ids)]

# Count the number of prescriptions filled by each pharmacy for these patients
pharmacy_counts = filtered_data.groupby('pharmacy').size()

# Display the count of prescriptions for each pharmacy
print(pharmacy_counts)
#%%
total_prescriptions = pharmacy_counts.sum()

# Calculate the percentage of total prescriptions for each pharmacy and print
pharmacy_percentage = (pharmacy_counts / total_prescriptions) * 100

# Display the percentage of total prescriptions for each pharmacy
print("Percentage of total prescriptions filled by each pharmacy:")
for pharmacy_id, percentage in pharmacy_percentage.items():
    print(f"Pharmacy {pharmacy_id}: {percentage:.2f}%")
#%%
# overall distribution of pharmacies
pharmacy_counts_overall = data.groupby('pharmacy').size()

# Display the count of prescriptions for each pharmacy
print(pharmacy_counts_overall)
#%%
total_prescriptions_overall = pharmacy_counts_overall.sum()

# Calculate the percentage of total prescriptions for each pharmacy and print
pharmacy_percentage_overall = (pharmacy_counts_overall / total_prescriptions_overall) * 100

# Display the percentage of total prescriptions for each pharmacy
print("Percentage of total prescriptions filled by each pharmacy:")
for pharmacy_id, percentage in pharmacy_percentage_overall.items():
    print(f"Pharmacy {pharmacy_id}: {percentage:.2f}%")

#%%
payer_type_counts = filtered_data.groupby('primary_payer_type').size()

# Display the count of prescriptions for each pharmacy
print(payer_type_counts)
#%%
payer_type_prescriptions = payer_type_counts.sum()

# Calculate the percentage of total prescriptions for each pharmacy and print
payer_type_percentage = (payer_type_counts / payer_type_prescriptions) * 100

# Display the percentage of total prescriptions for each pharmacy
print("Percentage of total prescriptions filled by each payer_type:")
for payer_type_id, percentage in payer_type_percentage.items():
    print(f"Payer_type {payer_type_id}: {percentage:.2f}%")
#%%
# overall distribution of pharmacies
payer_type_counts_overall = data.groupby('primary_payer_type').size()

# Display the count of prescriptions for each pharmacy
print(payer_type_counts_overall)
#%%
total_payer_type_overall = payer_type_counts_overall.sum()

# Calculate the percentage of total prescriptions for each pharmacy and print
payer_type_percentage_overall = (payer_type_counts_overall / total_payer_type_overall) * 100

# Display the percentage of total prescriptions for each pharmacy
print("Percentage of total prescriptions filled by each payer_type:")
for payer_type_id, percentage in payer_type_percentage_overall.items():
    print(f"Payer_type {payer_type_id}: {percentage:.2f}%")

#%%
# patients who have completed shifted from Brand A and never returned

# Filter the original DataFrame to include only the patients who shifted back to Brand A
df_patients_shift_stopped_brandA = data[data['patient_id'].isin(valid_switched_patient_ids)]
display(df_patients_shift_stopped_brandA)
#%%

#%%

#%%
# patients who have shifted from Brand A and still were receiving brand A
df_patients_shift_still_brandA = data[data['patient_id'].isin(shifted_patient_ids)]
display(df_patients_shift_still_brandA)
#%%
shifted_patient_ids_set = set(shifted_patient_ids)
valid_switched_patient_ids_set = set(valid_switched_patient_ids)
# Identify patients in shifted_patient_ids but not in valid_switched_patient_ids
patients_still_a = shifted_patient_ids_set - valid_switched_patient_ids_set
# Convert the result back to a list if needed
patients_still_a_list = list(patients_still_a)
df_patients_shift_only_kept_brandA_and_didnt_stop_unique = data[data['patient_id'].isin(patients_still_a_list)]
#%%
display(df_patients_shift_only_kept_brandA_and_didnt_stop_unique)
#%%
# Starting with Brand A: The patient must initially be prescribed Brand A.
# Shifting to Other Brands (B, C, or D): After starting with Brand A, the patient must shift to one of the other brands.
# Switching Back to Brand A: After shifting to another brand, the patient must then switch back to Brand A.
# Never Shifting Again: After returning to Brand A, the patient must not shift to any other brand again.

# No patients, so they were permanently lost
def identify_switch_back_and_stay_with_a(group):
    shifted = False  # Tracks if the patient has shifted to another brand
    switched_back_to_a = False  # Tracks if the patient switched back to Brand A after shifting
    stayed_with_a = True  # Assumes the patient stayed with A after switching back until proven otherwise
    prescribed_a = False  # Tracks if Brand A was initially prescribed

    for index, row in group.iterrows():
        if row['brand'] == 'A':
            if not prescribed_a:
                # Mark the initial prescription of Brand A
                prescribed_a = True
            elif shifted:
                # If already shifted and Brand A is prescribed again, mark switch back to A
                switched_back_to_a = True
        elif prescribed_a and row['brand'] in ['B', 'C', 'D']:
            if not switched_back_to_a:
                # Note the shift to another brand after Brand A
                shifted = True
                stayed_with_a = False  # Reset stayed_with_a since a shift occurred after initial A
            else:
                # If they switched back to A but then shifted again, they didn't stay with A
                stayed_with_a = False
                break  # No need to check further, as they shifted again after returning to A

    # Check if all conditions are met: started with A, shifted, switched back to A, and stayed with A
    return switched_back_to_a and stayed_with_a

# Apply the function to each patient's group
patients_who_switched_back_and_stayed = sorted_prescriptions.groupby('patient_id').apply(identify_switch_back_and_stay_with_a)

# Filter to keep only patients who meet the criteria
valid_patients = patients_who_switched_back_and_stayed[patients_who_switched_back_and_stayed]

# Extract unique patient IDs who switched back to A and stayed with A
valid_patient_ids = valid_patients.index.tolist()

print(f"Patient IDs who started with A, shifted to other brands, switched back to A, and then never shifted again: {valid_patient_ids}")

#%%

# Filter the DataFrame for the specified patient IDs
filtered_df_for_valid_patients = data[data['patient_id'].isin(valid_switched_patient_ids)]

# Extract the unique hcp_id associated with these patients
unique_hcp_ids_for_valid_patients = filtered_df_for_valid_patients['hcp_id'].unique()

# HCP_Id for patients who started taking another brand
print(f"Unique HCP IDs for the specified patients: {unique_hcp_ids_for_valid_patients.tolist()}")
#%%
# stopped taking drugs all together, after taking brand A
def patients_stopped_after_a(group):
    prescribed_a = False
    stopped_completely = False
    for index, row in group.iterrows():
        if row['brand'] == 'A':
            prescribed_a = True
        # Check for any activity after Brand A
        if prescribed_a and index > group[group['brand'] == 'A'].index.max():
            return False  # Found activity after Brand A, so not stopped
    return prescribed_a  # Return True if prescribed A and no activity afterward

# Apply the function to each patient's group
patients_stopped_after_a_result = sorted_prescriptions.groupby('patient_id').apply(patients_stopped_after_a)

# Filter to keep only patients who stopped after Brand A
stopped_after_a_patient_ids = patients_stopped_after_a_result[patients_stopped_after_a_result].index.tolist()

print(f"Patient IDs who stopped taking any drugs after taking Brand A: {stopped_after_a_patient_ids}")

#%%
len(stopped_after_a_patient_ids)
#%%
# second function to not only track patients who stopped taking any drugs after starting with Brand A and
# then started again but also identify which brand(s) they started taking upon resuming,

def patients_started_with_a_and_restarted_with_brands(group):
    started_with_a = False  # Flag to check if started with Brand A
    last_seen = None  # Track the last seen prescription date
    brands_after_gap = []  # Track brands after a significant gap

    # Ensure 'ship_date' is in datetime format for comparison
    group['ship_date'] = pd.to_datetime(group['ship_date'].dt.to_timestamp())

    for index, row in group.iterrows():
        if row['brand'] == 'A' and not started_with_a:
            # Mark that the patient started with Brand A
            started_with_a = True
            last_seen = row['ship_date']
            continue

        if started_with_a:  # Proceed only if the patient started with Brand A
            if last_seen is not None:
                # Calculate the gap between the current and last prescription
                gap = (row['ship_date'] - last_seen).days

                if gap > 30:  # Assuming a 30-day gap signifies stopping and starting again
                    brands_after_gap.append(row['brand'])

            # Update last_seen with the current prescription's date
            last_seen = row['ship_date']

    return brands_after_gap if brands_after_gap else None

# Apply the function to each patient's group
patients_restarted_with_brands_result = sorted_prescriptions.groupby('patient_id').apply(patients_started_with_a_and_restarted_with_brands)

# Filter out the groups with no restart (None values)
patients_restarted_with_brands = patients_restarted_with_brands_result.dropna()

# Create a dictionary mapping patient IDs to the brands they restarted with
restarted_brands_by_patient = patients_restarted_with_brands.to_dict()

print(f"Patient IDs with brands they restarted with after stopping, having started with Brand A: {restarted_brands_by_patient}")

#%%
#len(restarted_brands_by_patient) # number of key value pairs
len(restarted_brands_by_patient)
#%%
# Initialize a dictionary to count occurrences of each brand
brand_count = {}

# Iterate through the dictionary to count each brand
for brands in restarted_brands_by_patient.values():
    for brand in brands:
        if brand in brand_count:
            brand_count[brand] += 1
        else:
            brand_count[brand] = 1

brand_count
#%%
brand_patients = {}

# Assume restarted_brands_by_patient includes patient IDs as keys
for patient_id, brands in restarted_brands_by_patient.items():
    for brand in brands:
        if brand not in brand_patients:
            brand_patients[brand] = set()  # Initialize with an empty set
        brand_patients[brand].add(patient_id)  # Add patient ID to the set for the brand

# Now, calculate the count per brand and unique patient count
brand_count = {brand: len(patients) for brand, patients in brand_patients.items()}
unique_patient_count = len(set.union(*brand_patients.values()))  # Total unique patients across all brands
#%%
brand_count
# %% This code will give me for each month's 'shipment scheduled' for a patient, the days difference for when was the
# latest referral that was gotten for that patient. And then you average it for each month. So for each month,
# you'll get an average number of days it took for the patient to get their drug scheduled.
data_brand_a['status_dt'] = pd.to_datetime(data_brand_a['status_dt'])

# Filter for 'referral' and 'shipment scheduled' statuses
relevant_statuses = data_brand_a[data_brand_a['status_detail'].isin(['referral', 'shipment scheduled'])]

# Sort data by patient and date
sorted_data = relevant_statuses.sort_values(by=['patient_id', 'status_dt'])

# Initialize a DataFrame to hold average times for each month
average_times = []

# Loop through each month in the range from Jan 2021 to Sep 2022
for month_end in pd.date_range(start='2021-01-01', end='2022-09-30', freq='M'):
    # The period we're looking at
    month_period = month_end.to_period('M')

    # Filter shipments for the current month
    shipments_this_month = sorted_data[(sorted_data['status_detail'] == 'shipment scheduled') & (sorted_data['status_dt'].dt.to_period('M') == month_period)]

    time_diffs = []  # To store time differences for this month

    # For each shipment this month, find the earliest referral
    for _, shipment in shipments_this_month.iterrows():
        patient_id = shipment['patient_id']
        shipment_date = shipment['status_dt']

        # Get all referrals for this patient before this shipment
        referrals = sorted_data[(sorted_data['patient_id'] == patient_id) &
                                (sorted_data['status_detail'] == 'referral') &
                                (sorted_data['status_dt'] < shipment_date)]

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