import pandas as pd
import re
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

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
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_crossref(doi):
    base_url = f"https://api.crossref.org/works/{doi}"
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            data = response.json().get('message', {})
            year = data.get('issued', {}).get('date-parts', [[None]])[0][0]
            citations = data.get('is-referenced-by-count', 0)
            source_journal_list = data.get('container-title', [])
            source_journal = source_journal_list[0] if source_journal_list else None
            return year, citations, source_journal
        else:
            return None, None, None
    except Exception as e:
        print(f"Error querying DOI {doi}: {e}")
        return None, None, None


def query_metadata(data, output_path):
    
    def process_row(row):

        metadata = pd.Series({
            'doi': None,
            'Year': None,
            'Citations': None,
            'source_journal': None
        })

        # Extract DOI from the source link if the link is human annotated
        link = row['source']
        dois = extract_doi(link)
        if dois != []:
            for doi in dois:
                year, citations, journal = query_crossref(doi)
                if year or citations:
                    metadata = pd.Series({
                        'doi': doi,
                        'Year': year,
                        'Citations': citations,
                        'source_journal': journal
                    })
                else:
                    print(f'No references found for {doi}')
        else:
            print(f'DOI could not be extracted for {dois}')

        # Extract DOI from the source link if the link is cleanly generated: 1 doi per question
        # doi = row['source']
        # year, citations = query_crossref(doi)
        # if year or citations:
        #     metadata = pd.Series({
        #         'doi': doi,
        #         'Year': year,
        #         'Citations': citations
        #     })
        # else:
        #     print(f'No references found for {doi}')
        
        return metadata
        
    tqdm.pandas(desc="Adding metadata")
    new_columns = data.progress_apply(process_row, axis=1)
    
    data = pd.concat([data, new_columns], axis=1)

    data.to_json(output_path, orient='records', indent=2)
    print(f"Processed data saved to {output_path}") 


def main():
    # Load data
    # file_path = 'data/normalized.json'
    # data = pd.read_json(file_path)
    file_path = 'data\expert_mobi.json'
    data = pd.read_json(file_path)

    output_path = 'data/doi/journal_expertMoBi.json'
    query_metadata(data,output_path)


if __name__ == '__main__':
    main()

