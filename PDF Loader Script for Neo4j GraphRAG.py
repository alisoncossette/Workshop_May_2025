# PDF Loader Script for Neo4j GraphRAG
# This script focuses solely on loading PDF files into Neo4j

from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from dotenv import load_dotenv
import os
import time
import asyncio
import glob
import csv
import sys
import json
import re
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def conform(obj):
    """
    Recursively replace all 'properties': [] with 'properties': {} in dicts/lists.
    Additionally, ensure output conforms to ERExtractionTemplate expectations:
    - Top-level must be a dict with 'entities' and 'relations' as lists.
    - Add missing keys if absent.
    - Remove extraneous keys.
    - Fix common type errors.
    """
    # First, recursively fix 'properties': []
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "properties" and v == []:
                new_obj[k] = {}
            else:
                new_obj[k] = conform(v)
        # ERExtractionTemplate compatibility: enforce top-level structure
        # Only apply at the top level (not recursively)
        if set(new_obj.keys()) & {"entities", "relations"}:
            # Remove extraneous keys
            allowed_keys = {"entities", "relations"}
            new_obj = {k: v for k, v in new_obj.items() if k in allowed_keys}
            # Add missing keys
            if "entities" not in new_obj or not isinstance(new_obj["entities"], list):
                new_obj["entities"] = []
            if "relations" not in new_obj or not isinstance(new_obj["relations"], list):
                new_obj["relations"] = []
        return new_obj
    elif isinstance(obj, list):
        return [conform(item) for item in obj]
    else:
        return obj

# --- Utility: Always use conformed LLM output before validation or pipeline ingestion ---
def safe_json_loads_and_conform(content):
    import json
    try:
        return conform(json.loads(content))
    except Exception as e:
        print("[DEBUG] Could not conform or parse LLM output:", e)
        return None

# --- Example usage in pipeline (replace everywhere LLM output is parsed) ---
# Instead of: data = json.loads(llm_output)
# Use: data = safe_json_loads_and_conform(llm_output)

# (If pipeline expects dict, patch pipeline or hook to use conformed output)

# --- Neo4j Connection ---
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("Using OpenAI cloud backend")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# --- Load Allowed Company Names ---
company_csv_path = os.path.join("data", "Company_Filings.csv")

def load_company_names(csv_path):
    """Load exact company names from the Company_Filings.csv file."""
    names = set()
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row.get('name')
            if name:
                names.add(name.strip())
    return sorted(names)

allowed_company_names = load_company_names(company_csv_path)
print("Allowed company names:", allowed_company_names)

# --- Robust normalization for company name matching and automapping ---
def normalize_name(name):
    import re
    return re.sub(r'[^A-Z0-9 ]', '', name.upper().strip())

# --- Post-processing: Filter only allowed company names ---
allowed_set = set(name.upper() for name in allowed_company_names)  # Case-insensitive match

def filter_entities_and_log(parsed, allowed_set, automap_company_name=None):
    filtered = []
    for e in parsed.get('entities', []):
        raw_name = e.get('properties', {}).get('name', '')
        name = normalize_name(raw_name)
        allowed_normalized = set(normalize_name(n) for n in allowed_set)
        # Auto-map generic names if automap_company_name is provided
        if e.get('label') == 'Company' and name in {"COMPANY", "THE COMPANY", "REGISTRANT"}:
            if automap_company_name:
                print(f"[AUTOMAP] Replacing generic company '{raw_name}' with '{automap_company_name}'")
                e['properties']['name'] = automap_company_name
                filtered.append(e)
            else:
                print(f"[FILTERED] Dropped company: {raw_name}")
        elif e.get('label') == 'Company' and name not in allowed_normalized:
            print(f"[FILTERED] Dropped company: {raw_name}")
        else:
            # Always set to canonical allowed name if match found
            for allowed in allowed_set:
                if name == normalize_name(allowed):
                    e['properties']['name'] = allowed
            filtered.append(e)
    parsed['entities'] = filtered
    return parsed

# --- Helper to infer automap company name for a file (single-company context) ---
def infer_company_name_from_filename(filename, allowed_set):
    import re
    fname = normalize_name(filename)
    for cname in allowed_set:
        cname_clean = normalize_name(cname)
        if cname_clean in fname:
            return cname
    if len(allowed_set) == 1:
        return list(allowed_set)[0]
    return None

# --- Custom ERExtractionTemplate for KG Builder ---
joined_names = '\n'.join(f"- {name}" for name in allowed_company_names)
company_instruction = (
    "Extract only information about the following companies. "
    "If a company is mentioned but is not in this list, ignore it. "
    "When extracting, the company name must match exactly as shown below. "
    "Do not generate or include any company not on this list or an alternate name for any company on this list. "
    "ONLY USE THE COMPANY NAME EXACTLY AS SHOWN IN THE LIST. "
    "If the text refers to 'the Company', 'the Registrant', or uses a pronoun or generic phrase instead of a company name, you MUST look up and use the exact company name from the allowed list based on context (such as the file being processed). "
    "UNDER NO CIRCUMSTANCES should you output 'the Company', 'the Registrant', or any generic phrase as a company name. Only use the exact allowed company name.\n\n"
    f"Allowed Companies (match exactly):\n{joined_names}\n\n"
    "\nIMPORTANT: When returning the 'properties' field for any entity or relationship, always use a JSON object (e.g., {{}}), even if there are no properties. Never use an empty array ([]). If there are no properties, return 'properties': {{}}.\n"
)
custom_template = company_instruction + ERExtractionTemplate.DEFAULT_TEMPLATE
prompt_template = ERExtractionTemplate(template=custom_template)

# --- Initialize LLM and Embeddings ---
embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = OpenAILLM(
    model_name="gpt-4o",
    api_key=OPENAI_API_KEY
)

dimensions = 1536

# --- Patch OpenAILLM.ainvoke to print all outputs ---
original_llm_ainvoke = llm.ainvoke
async def debug_llm_ainvoke(*args, **kwargs):
    result = await original_llm_ainvoke(*args, **kwargs)
    import json
    content = None
    if isinstance(result, dict) and 'content' in result:
        content = result['content']
    elif hasattr(result, 'content'):
        content = result.content
    elif isinstance(result, str):
        content = result
    if content:
        try:
            conformed = safe_json_loads_and_conform(content)
            print("[DEBUG] LLM Output (conformed):", conformed)
        except Exception as ce:
            print("[DEBUG] Could not conform LLM output in debug_llm_ainvoke:", ce)
    return result

# --- Define Node Labels and Relationship Types ---
entities = [
    {"label": "Executive", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "Product", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "FinancialMetric", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "RiskFactor", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "StockType", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "Transaction", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "TimePeriod", "properties": [{"name": "name", "type": "STRING"}]},
    {"label": "Company", "properties": [{"name": "name", "type": "STRING"}]}
]
relations = [
    {"label": "HAS_METRIC"},
    {"label": "FACES_RISK"},
    {"label": "ISSUED_STOCK"},
    {"label": "MENTIONS"}
]   

# --- DEBUG: Print Final Prompt ---
print("\n[DEBUG] --- FINAL PROMPT PASSED TO LLM (custom template) ---\n")
print(prompt_template.format(
    text="EXAMPLE TEXT",
    schema=entities,
    examples=[]
))
print("\n[DEBUG] --- END PROMPT ---\n")

# --- Minimal Template Test ---
def get_minimal_pipeline():
    from neo4j_graphrag.generation.prompts import ERExtractionTemplate
    print("\n[DEBUG] --- FINAL PROMPT PASSED TO LLM (default template) ---\n")
    print(ERExtractionTemplate().format(
        text="EXAMPLE TEXT",
        schema=entities,
        examples=[]
    ))
    print("\n[DEBUG] --- END PROMPT ---\n")
    return SimpleKGPipeline(
        driver=driver,
        llm=llm,
        embedder=embedder,
        entities=entities,
        relations=relations,
        prompt_template=ERExtractionTemplate(),
        enforce_schema="STRICT"
    )

# --- Initialize and Run the Pipeline ---
print("Entities being passed to pipeline:", entities)
print("Type of entities:", type(entities))

try:
    pipeline = SimpleKGPipeline(
        driver=driver,
        llm=llm,
        embedder=embedder,
        entities=entities,
        relations=relations,
        prompt_template=prompt_template,
        enforce_schema="STRICT"
    )
    # --- DEBUG: Run minimal pipeline for first file only ---
    form10k_dir = "data/form10k-sample"
    pdf_files = glob.glob(form10k_dir + "/*.pdf")
    if pdf_files:
        print("\n[DEBUG] --- RUNNING MINIMAL TEMPLATE PIPELINE ON FIRST FILE ONLY ---\n")
        minimal_pipeline = get_minimal_pipeline()
        try:
            asyncio.run(minimal_pipeline.run_async(file_path=pdf_files[0]))
        except Exception as e:
            print("[DEBUG] Minimal template pipeline error:", e)
            if hasattr(e, 'llm_output'):
                try:
                    conformed = safe_json_loads_and_conform(getattr(e, 'llm_output'))
                    print("[DEBUG] LLM Output (conformed):", conformed)
                except Exception as ce:
                    print("[DEBUG] Could not conform LLM output:", ce)
    print("\n[DEBUG] --- END MINIMAL TEMPLATE PIPELINE TEST ---\n")
except Exception as e:
    print("Pipeline initialization failed:", e)
    raise

# --- Enhanced Debug Logging and Schema Validation for LLM Output ---
def validate_llm_schema(obj):
    # Example: expects a dict with 'entities' (list) and 'relations' (list)
    # Adjust this function to match your expected schema
    if not isinstance(obj, dict):
        return False
    if 'entities' not in obj or 'relations' not in obj:
        return False
    if not isinstance(obj['entities'], list) or not isinstance(obj['relations'], list):
        return False
    return True

# --- Pipeline Patch: Always use conformed LLM output ---
# Patch or wrap all usages of json.loads(llm_output) to safe_json_loads_and_conform(llm_output)

# 1. Minimal template pipeline test (if used):
# Already handled in exception handler with safe_json_loads_and_conform

# 2. Main pipeline run: Patch pipeline.run_async or wherever LLM output is parsed
# If you have direct parsing of LLM output, change:
# data = json.loads(llm_output)
# to:
# data = safe_json_loads_and_conform(llm_output)

# 3. If pipeline is a class, patch methods that parse LLM output
# For example, if there is a method like parse_llm_output, update it:
# def parse_llm_output(llm_output):
#     return safe_json_loads_and_conform(llm_output)

# 4. If you use a utility function for parsing, update it to call safe_json_loads_and_conform

# 5. If you use run_pipeline_on_file, ensure it uses safe_json_loads_and_conform

async def run_pipeline_on_file(file_path, pipeline):
    try:
        await pipeline.run_async(file_path=file_path)
    except Exception as e:
        print(f"[DEBUG] Error running pipeline on {file_path}: {e}")
        # Attempt to print LLM output if available
        if hasattr(e, 'llm_output'):
            try:
                conformed = safe_json_loads_and_conform(getattr(e, 'llm_output'))
                print("[DEBUG] LLM Output (conformed):", conformed)
            except Exception as ce:
                print("[DEBUG] Could not conform LLM output:", ce)
        raise

# --- Async Pipeline Run Example ---
async def run_pipeline_on_file(file_path, pipeline):
    try:
        await pipeline.run_async(file_path=file_path)
    except Exception as e:
        print(f"[DEBUG] Error running pipeline on {file_path}: {e}")
        # Attempt to print LLM output if available
        if hasattr(e, 'llm_output'):
            try:
                conformed = safe_json_loads_and_conform(getattr(e, 'llm_output'))
                print("[DEBUG] LLM Output (conformed):", conformed)
            except Exception as ce:
                print("[DEBUG] Could not conform LLM output:", ce)
        raise

# --- After parsing/conforming LLM output, filter entities and log dropped companies ---
# This should be placed immediately after you parse/conform the LLM output for each chunk/file

# Run the pipeline on all files in data/form10k-sample before retrieval
form10k_dir = "data/form10k-sample"
pdf_files = glob.glob(form10k_dir + "/*.pdf")

print(f"\nFound {len(pdf_files)} PDF files to process")
for pdf_file in pdf_files:
    try:
        print(f"\nProcessing file: {pdf_file}")
        llm_output = None
        parsed = None
        try:
            result = asyncio.run(pipeline.run_async(file_path=pdf_file))
            llm_output = result.llm_output
            parsed = safe_json_loads_and_conform(llm_output)
            automap_name = infer_company_name_from_filename(pdf_file, allowed_set)
            parsed = filter_entities_and_log(parsed, allowed_set, automap_company_name=automap_name)  # Apply automapping here
        except Exception as e1:
            print(f"[DEBUG] JSON parse error for file {pdf_file}: {e1}\nRaw output: {llm_output}")
            try:
                parsed = conform(json.loads(llm_output))
                print(f"[DEBUG] Conformed output for file {pdf_file}: {parsed}")
                automap_name = infer_company_name_from_filename(pdf_file, allowed_set)
                parsed = filter_entities_and_log(parsed, allowed_set, automap_company_name=automap_name)  # Apply automapping here
            except Exception as e2:
                print(f"[DEBUG] Could not conform or parse LLM output for file {pdf_file}: {e2}")
        if parsed is not None:
            if not validate_llm_schema(parsed):
                print(f"[DEBUG] LLM output schema invalid for file {pdf_file}: {parsed}")
                # Optionally, skip or attempt to auto-correct here
                continue
        else:
            print(f"[DEBUG] LLM output could not be parsed or conformed for file {pdf_file}.")
            continue
        print(f"Successfully processed: {pdf_file}")
        time.sleep(21)  # Add a delay between requests to avoid hitting OpenAI's RPM limit
    except Exception as e:
        print(f"Error processing {pdf_file}: {str(e)}")
        # Print only conformed LLM output for debugging
        if hasattr(e, 'llm_output'):
            print(f"[DEBUG] LLM Output (raw) for file {pdf_file}: {getattr(e, 'llm_output', 'N/A')}")
            try:
                conformed = safe_json_loads_and_conform(getattr(e, 'llm_output'))
                print(f"[DEBUG] LLM Output (conformed) for file {pdf_file}: {conformed}")
            except Exception as ce:
                print("[DEBUG] Could not conform LLM output:", ce)
        print("Continuing with next file...")
        continue

print("\nPDF loading complete!")

from neo4j_graphrag.indexes import create_vector_index

create_vector_index(driver, name="text_embeddings", label="Chunk",
                    embedding_property="embedding", dimensions=1536, similarity_fn="cosine")
print("Successfully created vector index 'chunkEmbeddings'")
driver.close()

## -- Now head to http://console.neo4j.io and upload the structured data from the data folder
## -- Use the neo4j_importer_model.json as the model