import pandas as pd
import requests
import json
from openpyxl import Workbook

# Read the input Excel file
input_df = pd.read_excel('12 week and rerun.xlsx')

# Create a new workbook for output
wb = Workbook()
ws = wb.active
ws.append(['Course Name', 'NPTEL URL', 'Timeline', 'noc_courseid', 'Enrolled', 'Registered', 'Certified', 'Gold', 'Silver', 'Elite', 'Success', 'Participation', 'Toppers'])

# Function to get course name from API
def get_course_name(course_id):
    api_url = f'https://tools.nptel.ac.in/npteldata/downloads.php?id={course_id}'
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = json.loads(response.text)
            if data['message'] == 'Success' and 'data' in data and 'title' in data['data']:
                return data['data']['title']
    except Exception as e:
        print(f"Error fetching course name for ID {course_id}: {str(e)}")
    return 'Not Found'

# Iterate through each row in the input Excel
for index, row in input_df.iterrows():
    nptel_url = row['NPTEL URL']
    course_id = nptel_url.split('/')[-1]
    
    # Get course name
    course_name = get_course_name(course_id)
    
    # Make API request for stats
    api_url = f'https://tools.nptel.ac.in/npteldata/stats.php?id={course_id}'
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = json.loads(response.text)
            
            if data['message'] == 'Success' and 'data' in data:
                for course_data in data['data']:
                    if 'run_wise_stats' in course_data:
                        for run_stats in course_data['run_wise_stats']:
                            ws.append([
                                course_name,
                                nptel_url,
                                run_stats['Timeline'],
                                run_stats['noc_courseid'],
                                run_stats['Enrolled'],
                                run_stats['Registered'],
                                run_stats['Certified'],
                                run_stats['Gold'],
                                run_stats['Silver'],
                                run_stats['Elite'],
                                run_stats['Success'],
                                run_stats['Participation'],
                                run_stats['Toppers']
                            ])
        else:
            print(f"Failed to fetch stats for {nptel_url}")
    except Exception as e:
        print(f"Error processing {nptel_url}: {str(e)}")

# Save the output Excel file
wb.save('output_file.xlsx')
print("Output file has been created: output_file.xlsx")