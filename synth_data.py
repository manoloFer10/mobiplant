import os
import json
import re
from pathlib import Path
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai import types
from pydantic import BaseModel
import random
from tqdm import tqdm
import enum

# ------------------ Models ------------------
class Area(enum.Enum):
  GENE_REGULATION = 'GENE REGULATION'
  GENOME_AND_GENOMICS = 'GENOME AND GENOMICS'
  CELL = 'CELL BIOLOGY AND CELL SIGNALING'
  GROWTH_DEVELOPMENT = 'GROWTH AND DEVELOPMENT'
  HORMONES = 'HORMONES'
  PHYSIOLOGY_METABOLSIM = 'PHYSIOLOGY AND METABOLISM'
  EVOLUTION = 'EVOLUTION'
  BIOTECH = 'BIOTECHNOLOGY'
  ENVIRONMENT = 'ENVIRONMENT'
  #NULL = None


class MCQSchema(BaseModel):
    question: str
    correct_answer: str
    incorrect_answer1: str
    incorrect_answer2: str
    area: Optional[Area] = None  # Optional area field
    plant_species: str 

class FinalMCQSchema(BaseModel):
    question: str
    options: List[str]
    answer: int  # Index of correct answer in options
    source: str  # DOI from filename
    source_journal: str  # Parent folder name
    area: Optional[Area] = None
    plant_species: str

# ------------------ Processor ------------------
class JournalPDFProcessor:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.prompt = """**Objective**
Generate 5 independent multiple-choice questions (3 options each) from a publicated paper. Format them as a JSON array of objects with fields: question, correct_answer, incorrect_answer1, incorrect_answer2, area (optional), and plant_species (default: "non-specific").

**Step-by-Step Instructions**
- Identify Key Facts:

Read the whole text to understand the work and carefully extract 5 distinct facts (e.g., conclusions, unique claims).
If a fact involves a plant species, note its scientific name.
Assign an area (must be one from, gene regulation, genome and genomics, cell biology and signaling, growth and development, hormones, physiology and metabolism, evolution, biotechnology or environment) if applicable; otherwise, use null.

- Craft Unique Questions:

From the extracted facts, formulate questions that are clear, concise, and unambiguous:
    - The five resulting questions should be different from each other, resulting in a unique set of questions. 
    - The questions should NOT be pointing to the study (e.g. What happened in this study?, In the work done by XX, ... , Acording to the study/text ...). Instead, they should be phrased without mentioning the document (e.g. What proteins have been identified as molecular partners of the Arabidopsis lncRNA ASCO?).
    - The questions should not represent methods/techniques or specific experimental details (e.g. What is the relative timing of cell expansion versus cell division in ...,). Rather, they should focus on the main concepts or conclusions of the study.

Note: For plant-related facts: Explicitly mention the species in the question (e.g., “What adaptation does Zea mays use to…?”).

- Generate Options:

Correct Answer: Directly derive from the text.

Distractors: Create two incorrect options by altering key details from the correct answer (e.g., species names, quantities, causal relationships, gene names).

- Assign Metadata:

Tag area only if the question aligns with a clear discipline from the provided list, else use "null".

For plant species: Use exact scientific names (e.g., "Oryza sativa").

- Validate:

Once chosen, ensure questions are asked about specific conclusions or key concepts about the work. They should not be about methods or experimental details.
If any question does not meet the criteria, discard it and create a new one that does.
Confirm plant_species is "non-specific" unless explicitly tied to a species.

**Output Format**
Return a JSON array adhering to this schema:
{schema}
""".format(schema=MCQSchema.schema_json(indent=2))

    def process_pdf(self, pdf_path: Path, journal_name: str) -> List[FinalMCQSchema]:
        try:
            # Extract DOI from filename (format: "YYYY DOI.pdf")
            filename = pdf_path.stem
            if ' ' in filename:
                doi = filename.split(' ', 1)[1].strip()  # Get text after first space
            else:
                doi = filename  # Fallback for malformed names

            # Process PDF with Gemini
            response = self._query_gemini(pdf_path)
            
            return self._transform_response(
                response=response,
                doi=doi,
                journal=journal_name
            )
            
        except Exception as e:
            # Capture detailed error information
            error_msg = f"""
            Error processing {pdf_path.name}:
            - DOI: {doi}
            - Journal: {journal_name}
            - Error: {str(e)}
            """
            raise RuntimeError(error_msg) from e

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _query_gemini(self, pdf_path: Path) -> str:
        response = self.client.models.generate_content(
                model="gemini-2.5-pro-preview-03-25",
                contents=[
                    types.Part.from_bytes(
                        data=pdf_path.read_bytes(),
                        mime_type="application/pdf"
                    ),
                    self.prompt
                ],
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': list[MCQSchema],
                }
            )
        return response

    def _transform_response(self, response, doi: str, journal: str) -> List[FinalMCQSchema]:
        try:
            raw_mcqs = json.loads(response.text) # List(MCQSchema)
            if not isinstance(raw_mcqs, list):
                raise ValueError("Expected array in response")
            
            final_results = []
            for mcq in raw_mcqs:
                base_mcq = MCQSchema(**mcq)
                options = [
                    base_mcq.correct_answer,
                    base_mcq.incorrect_answer1,
                    base_mcq.incorrect_answer2
                ]
                
                # Store correct answer before shuffling
                correct_answer = base_mcq.correct_answer
                
                # Shuffle options to randomize position
                random.shuffle(options)
                
                # Find new index of correct answer
                try:
                    answer_index = options.index(correct_answer)
                except ValueError:
                    raise ValueError("Correct answer missing from options after shuffling")
                
                final_results.append(FinalMCQSchema(
                    question=base_mcq.question,
                    options=options,
                    answer=answer_index,
                    source= self._transform_doi(doi),
                    source_journal=journal,
                    area=base_mcq.area,
                    plant_species=base_mcq.plant_species
                ))
            
            return final_results
            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response")
        except Exception as e:
            raise ValueError(f"Data transformation failed: {str(e)}")
    
    def _transform_doi(self, doi: str) -> str:
        """Normalize DOI from filename"""          
        # Normalize DOI: replace hyphens with slashes
        normalized_doi = doi.strip().replace("_", "/")
        return normalized_doi

    def process_journal_folders(self, root_input: str, output_path: str):
        output_file = Path(output_path) / "filtered_questions.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load already processed entries
        processed = self._load_processed_entries(output_file)

        journal_dirs = list(Path(root_input).iterdir())
        with tqdm(journal_dirs, desc="Processing journals", unit="journal") as journal_pbar:
            for journal_dir in journal_pbar:
                if not journal_dir.is_dir():
                    continue
                
                journal_pbar.set_postfix({"journal": journal_dir.name})
                
                pdf_files = list(journal_dir.glob("*.pdf"))
                with tqdm(pdf_files, desc="PDFs in journal", leave=False, unit="file") as pdf_pbar:
                    for pdf_file in pdf_pbar:
                        # Check if already processed
                        doi = self._extract_doi(pdf_file.name)
                        journal_name = journal_dir.name
                        
                        if (journal_name, doi) in processed:
                            pdf_pbar.set_postfix_str(f"Skipped (already processed)")
                            pdf_pbar.update(1)
                            continue
                        
                        try:
                            results = self.process_pdf(pdf_file, journal_name)
                            self._append_results(results, output_file)
                            pdf_pbar.set_postfix({"success": len(results)})
                        except Exception as e:
                            print(f"\nSkipped {pdf_file.name}: {str(e)}")
                        finally:
                            pdf_pbar.update(1)

    def _load_processed_entries(self, output_file: Path) -> set:
        """Load already processed journal+doi combinations"""
        processed = set()
        
        if output_file.exists():
            with output_file.open("r", encoding="utf-8") as f:
                for line in tqdm(f, desc="Loading progress", unit=" entries"):
                    try:
                        data = json.loads(line)
                        processed.add((data["source_journal"], data["source"].replace("/","_")))
                    except json.JSONDecodeError:
                        print(f"Invalid JSON line: {line.strip()}")
                        continue
        return processed

    def _extract_doi(self, filename: str) -> str:
        """Extract DOI from filename with improved validation"""
        try:
            filename = filename.removesuffix('.pdf') # Remove .pdf extension
            # Split on first space and remove year
            parts = filename.split(" ", 1)
            return parts[1].strip() 
        except IndexError:
            # Fallback pattern for non-standard filenames
            match = re.search(r"\b10\.\d{4,}/[\w./-]+", filename)
            return match.group(0) if match else filename

    def _append_results(self, mcqs: List[FinalMCQSchema], output_file: Path):
        """Append results as JSON lines"""
        with output_file.open("a", encoding="utf-8") as f:
            for mcq in mcqs:
                f.write(mcq.model_dump_json() + "\n")

def get_keys():
    keys_path = Path('tokens.json')
    with keys_path.open('r', encoding='utf-8') as f:
        keys = json.load(f)
    return keys

# ------------------ Usage ------------------
if __name__ == "__main__":
    key = get_keys()['gemini']
    processor = JournalPDFProcessor(api_key=key)
    processor.process_journal_folders(
        root_input="data/paper_collection",
        output_path="data/synthetic_data"
    )
