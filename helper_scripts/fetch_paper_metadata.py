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
    return dois  

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
        
    tqdm.pandas(desc="Adding metadata")
    new_columns = data.progress_apply(process_row, axis=1)
    
    data = pd.concat([data, new_columns], axis=1)

    data.to_json(output_path, orient='records', indent=2)
    print(f"Processed data saved to {output_path}") 

def extractable(data):
    # Create DOI column using apply() with progress bar
    tqdm.pandas(desc="Extracting DOIs")
    data['doi'] = data['source'].progress_apply(lambda x: extract_doi(x) or None)
    
    # Save results
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

