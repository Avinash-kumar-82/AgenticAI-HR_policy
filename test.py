from src.backend.core.config import get_settings
from backend.services.ingestion import load_file,chunk_documents
# settings=get_settings()

from pathlib import Path
docs=load_file(Path("data/sample_kb/company_hr_handbook.md"))

chunked_docs=chunk_documents(docs)
print(len(chunked_docs))

print("======================================")
print(chunked_docs)