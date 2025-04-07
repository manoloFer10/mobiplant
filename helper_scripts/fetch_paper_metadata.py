# import pandas as pd
# import re
# import requests
# import time
# from tqdm import tqdm

# # Load data
# file_path = 'data\mcq_results_all_shuffles.csv'
# data = pd.read_csv(file_path)

# session = requests.Session()
# session.headers.update({
#     'User-Agent': 'get-dois/1.0'
# })

# def extract_doi(link):
#     doi_pattern = r'(?i)\b(?:doi\s*[:]?\s*)?(10[.]\d+\/[^\s&]*)'
#     match = re.search(doi_pattern, link)
#     cleaned_doi = None
#     if match:
#         cleaned_doi = re.sub(r'[^\w./-]+$', '', match.group(1)).strip()
#     return cleaned_doi

# # Function to query CrossRef API
# def query_crossref(doi):
#     base_url = f"https://api.crossref.org/works/{doi}"
#     try:
#         response = session.get(base_url, timeout=10)
#         if response.status_code == 200:
#             data = response.json().get('message', {})
#             year = data.get('issued', {}).get('date-parts', [[None]])[0][0]
#             citations = data.get('is-referenced-by-count', 0)
#             return year, citations
#         else:
#             return None, None
#     except Exception as e:
#         print(f"Error querying DOI {doi}: {e}")
#         return None, None

# # Process data with progress bar
# results = []
# for index, row in tqdm(data.iterrows(), total=data.shape[0]):
#     link = row['source']
#     doi = extract_doi(link)
#     if doi:
#         for doi in doi:
#             year, citations = query_crossref(doi)
#             results.append({
#                 'Original Source': link,
#                 'DOI': doi,
#                 'Year': year,
#                 'Citations': citations
#             })
#             time.sleep(0.1)  # Be polite with a small delay
#     else:
#         results.append({
#             'Original Source': link,
#             'DOI': None,
#             'Year': None,
#             'Citations': None
#         })

# # Save results to CSV
# output_path = 'data/paper_data.csv'
# results_df = pd.DataFrame(results)
# results_df.to_csv(output_path, index=False)

# print(f"Processed data saved to {output_path}")


import pandas as pd
import re
import requests
from tqdm import tqdm

# Function to extract DOIs from links
def extract_doi(link):
    doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
    matches = re.findall(doi_pattern, link)
    dois = []
    for doi in matches:
        if doi.endswith('.') or doi.endswith(';'):
            doi = doi[:-1]
        dois.append(doi)
    return dois  # Returns an empty list if no DOIs are found

# Function to query CrossRef API
def query_crossref(doi):
    base_url = f"https://api.crossref.org/works/{doi}"
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            data = response.json().get('message', {})
            year = data.get('issued', {}).get('date-parts', [[None]])[0][0]
            citations = data.get('is-referenced-by-count', 0)
            return year, citations
        else:
            return None, None
    except Exception as e:
        print(f"Error querying DOI {doi}: {e}")
        return None, None


def query_metadata(data):
    output_path = 'data/doi/processed_paper_data.json'
    
    # Define processing function for each row
    def process_row(row):
        link = row['source']
        dois = extract_doi(link)

        metadata = pd.Series({
            'doi': None,
            'Year': None,
            'Citations': None
        })
        if dois != []:
            for doi in dois:
                year, citations = query_crossref(doi)
                if year or citations:
                    metadata = pd.Series({
                        'doi': doi,
                        'Year': year,
                        'Citations': citations
                    })
                else:
                    print(f'No references found for {doi}')
        else:
            print(f'DOI could not be extracted for {link}')
        return metadata
        
    # Apply with progress bar
    tqdm.pandas(desc="Adding metadata")
    new_columns = data.progress_apply(process_row, axis=1)
    
    # Merge new columns with original DataFrame
    data = pd.concat([data, new_columns], axis=1)
    
    # Save updated DataFrame
    data.to_json(output_path, orient='records', indent=2)
    print(f"Processed data saved to {output_path}") 

def extractable(data):
    # Create DOI column using apply() with progress bar
    tqdm.pandas(desc="Extracting DOIs")
    data['doi'] = data['source'].progress_apply(lambda x: extract_doi(x) or None)
    
    # Save results to CSV
    output_path = 'data/doi/extractable_dois.json'
    data.to_json(output_path, index=False, orient='records', indent=2)
    
    print(f"Processed data saved to {output_path}")
    return data

def main():
    # Load data
    file_path = 'data/normalized.json'
    data = pd.read_json(file_path)

    query_metadata(data)


if __name__ == '__main__':
    main()

