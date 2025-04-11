from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import pandas as pd

url = 'https://www.google.com/search?q=operations%20research%20analyst%20jobs%20in%20usa&rlz=1C5MACD_enUS1048US1048&oq=operations%20research%20analyst&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MgYIARBFGDsyBggCEEUYOzIGCAMQRRhAMgYIBBBFGDwyBggFEEUYPDIGCAYQRRg80gEJMTQ5ODVqMGoxqAIAsAIB&sourceid=chrome&ie=UTF-8&sei=HsavZ5upBr3k5NoPyLLhsQI&jbr=sep:0&udm=8&ved=2ahUKEwjI_KnGncSLAxVXKVkFHeUHMigQ3L8LegQIMhAN'

s = Service("/Users/mogana/Downloads/chromedriver-mac-arm64/chromedriver")
driver = webdriver.Chrome(service=s)

def extract():
    driver.get(url)
    # Scroll down multiple times to load more results
    for _ in range(10):  # Adjust number of scrolls
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(2)  # Allow time for new jobs to load
        # Get the updated HTML after scrolling
    soup = BeautifulSoup(driver.page_source, "html.parser")

    return soup #r.status_code



def transform(soup):
    divs = soup.find_all('a', class_='MQUd2b')
    print(f"Total jobs found: {len(divs)}\n")
    joblist = []  # Initialize the joblist to store job details
    
    for job in divs:
        # Initialize each job's details to default values
        title, company, location, salary, dental_insurance, health_insurance, paid_time_off = None, None, None, "Not Provided", None, None, None
        
        try:
            # Extract Job Title
            title = job.find("div", class_="tNxQIb PUpOsf").text.strip()
        except AttributeError:
            title = "Not Provided"
        
        try:
            # Extract Company
            company = job.find("div", class_="wHYlTd MKCbgd a3jPc").text.strip()
        except AttributeError:
            company = "Not Provided"
        
        try:
            # Extract Location
            location = job.find("div", class_="wHYlTd FqK3wc MKCbgd").text.strip()
            location = location.split("•")[0].strip() if location else "Not Provided"
        except AttributeError:
            location = "Not Provided"
        
        try:
            # Extract Salary
            salary_element = job.find("div", class_="K3eUK QZEeP").find("span", class_="Yf9oye")
            salary = salary_element.text.strip() if salary_element else "Not Provided"
        except AttributeError:
            salary = "Not Provided"
        
        try:
            # Extract benefits inside <div class="HvHIEc">
            benefits = [benefit.find_all("span")[1].text.strip() for benefit in job.find_all("div", class_="HvHIEc")]
            
            # Assign each benefit to a variable if available
            dental_insurance = "Yes" if "Dental insurance" in benefits else None
            health_insurance = "Yes" if "Health insurance" in benefits else None
            paid_time_off = "Yes" if "Paid time off" in benefits else None
        except AttributeError:
            benefits = []
            dental_insurance = None
            health_insurance = None
            paid_time_off = None
        
        # Create job dictionary
        job_details = {
            'title': title,
            'company': company,
            'location': location,
            'salary': salary,
            'dental_insurance': dental_insurance,
            'health_insurance': health_insurance,
            'paid_time_off': paid_time_off
        }
        
        joblist.append(job_details)

    return joblist  # Return the joblist after processing all jobs


# Sample usage (assuming extract() function is defined elsewhere):
c = extract()  # This should return the soup object for scraping
job_list = transform(c)  # Process the soup object and extract job details
print(job_list)  # Print the final list of jobs
print(len(job_list))

df = pd.DataFrame(job_list)
print(df.head())
df.to_csv('OpGoogleSalaries.csv')