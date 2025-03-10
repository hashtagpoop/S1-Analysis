import sec_parser as sp
import requests
import re
import requests
import re
from sec_downloader import Downloader

COMPANY_NAME = "MyCompany"
COMPANY_EMAIL = "mycompany@example.com"
SEC_BASE_URL = "https://efts.sec.gov/LATEST/search-index"

def create_edgar_url(cik, filing_id):
    file_identifer, file_name = filing_id.split(":")
    file_identifer = file_identifer.replace("-", "")
    edgar_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{file_identifer}/{file_name}"
    return edgar_url

# TODO: fix this so we can just pass in the form type itself
# and download the one that we want.
def download_filing(input_file: str, form: str = "S-1"):
    # The company name and email address just need to be valid formats,
    # but doesn't have to be an actual company or email.
    dl = Downloader(COMPANY_NAME, COMPANY_EMAIL)
    s1_content = dl.get_filing_html(query=input_file)
    return s1_content

# TODO: maybe deprecate this since we don't need this to get the S1 filings
# anymore. Rather we want to be able to pull from the SEC API downloader
# there is a metadata and form downloader api.
# Gets S1 filings from the SEC EDGAR database.
def get_s1_filings(start_date: str, end_date: str):
    base_url = SEC_BASE_URL
    
    start_from = 0
    size = 100
    all_filings = []

    headers = {
        "User-Agent": f"{COMPANY_NAME}/1.0 ({COMPANY_EMAIL})",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    initial_url = f"{base_url}?dateRange=custom&category=custom&startdt={start_date}&enddt={end_date}&forms=S-1&page=1&from=0"
    response = requests.get(initial_url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        total_hits = data.get("hits", {}).get("total", {}).get("value", 0)
        print(f"Total S-1 filings found: {total_hits}")
        
        while start_from < total_hits:
            url = f"{base_url}?dateRange=custom&category=custom&startdt={start_date}&enddt={end_date}&forms=S-1&page={start_from // size + 1}&from={start_from}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                filings = data.get("hits", {}).get("hits", [])
                
                for filing in filings:
                    source_data = filing.get("_source")
                    # There should only be one display name per filing, hence the [0].
                    display_names = source_data.get("display_names")[0]
                    # Use regex to extract CIK from the display name.
                    cik_match = re.search(r"CIK (\d+)", display_names)
                    cik = cik_match.group(1) if cik_match else None
    
                    # Use regex to extract CIK from the string
                    cik_match = re.search(r"CIK (\d+)", display_names)
                    cik = cik_match.group(1) if cik_match else None
                    all_filings.append({
                        "filing_id": filing.get("_id"),
                        "display_names": source_data.get("display_names"),
                        "file_date": source_data.get("file_date"),
                        "file_type": source_data.get("file_type"),
                        "biz_locations": source_data.get("biz_locations"),
                        "sequence": source_data.get("sequence"),
                        "inc_states": source_data.get("inc_states"),
                        "cik": cik
                    })
                
                start_from += size 
            else:
                print(f"Error: Received status code {response.status_code}")
                print(response.text)
                break
    else:
        print(f"Error: Received status code {response.status_code}")
        print(response.text)
    
    return all_filings