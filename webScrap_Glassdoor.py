from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
from bs4 import BeautifulSoup

# Set up the driver
url = 'https://www.glassdoor.com/Job/united-states-it-project-manager-jobs-SRCH_IL.0,13_IN1_KO14,32.htm?remoteWorkType=1&seniorityType=midseniorlevel'

s = Service("/Users/mogana/Downloads/chromedriver-mac-arm64/chromedriver")
driver = webdriver.Chrome(service=s)
driver.get(url)

def extract_jobs(soup):
    job_list = []

    job_listings = soup.find_all('li', class_='JobsList_jobListItem__wjTHv')
    print(f"Total jobs extracted: {len(job_listings)}\n")

    for job in job_listings:
        title = company = location = salary = "N/A"

        # Extract Company Name
        company_elem = job.find("div", class_="EmployerProfile_compactEmployerName__9MGcV")
        if company_elem:
            company = company_elem.text.strip()

        # Extract Job Title
        title_elem = job.find("a", class_="JobCard_jobTitle__GLyJ1")
        if title_elem:
            title = title_elem.text.strip()


        # Extract Job Location
        location_elem = job.find("div", class_="JobCard_location__Ds1fM")
        if location_elem:
            location = location_elem.text.strip()

        # Extract Salary (if available)
        salary_elem = job.find("div", class_="JobCard_salaryEstimate__QpbTW")
        if salary_elem:
            salary = salary_elem.text.strip()

        # Store in dictionary
        job_list.append({
            'Title': title,
            'Company': company,
            'Location': location,
            'Salary': salary
        })

    return job_list

# Update the load_all_jobs function to pass soup to extract_jobs
def load_all_jobs():
    while True:
        try:
            load_more_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//button[@data-test="load-more"]'))
            )
            driver.execute_script("arguments[0].click();", load_more_button)
            time.sleep(3)  # Allow jobs to load
            print("Clicked 'Show More Jobs'...")
        except:
            print("No more 'Show More Jobs' button found.")
            break  # Exit loop when button is no longer available

    # Once all jobs are loaded, extract and save the data
    soup = BeautifulSoup(driver.page_source, "html.parser")  # Parse page source
    job_data = extract_jobs(soup)  # Pass soup to extract_jobs function
    return job_data

# Load all jobs and extract job data
job_data = load_all_jobs()

# Save results to CSV
df = pd.DataFrame(job_data)
df.to_csv(f'Glassdoor_Jobs_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
print(df.head())

# Close the driver
driver.quit()
