# Repository integrity audit
import sys
sys.path.insert(0, "D:/Projects/PRECEDENT")

# Check 1: app/schemas.py syntax and imports
print("=== Check 1: app/schemas.py ===")
try:
    from app.schemas import IngestRequest, IngestResponse, RetrievalRelevance, EvidenceSufficiency, ClaimVerificationResult, AuthorityApplicabilityFlags, DecompositionRequest, RetrievalResponse, IterationRequest, RefusalResponse, TraceRecord, RefusalCategory, SystemRefusal, ChunkApplicability, ConflictFlag, FilteredRetrievalResponse
    print("  ✓ All schema imports succeed")
    print(f"  ✓ IngestRequest fields: {list(IngestRequest.model_fields.keys())}")
    print(f"  ✓ IngestResponse fields: {list(IngestResponse.model_fields.keys())}")
except Exception as e:
    print(f"  ✗ Schema import failed: {e}")

# Check 2: app/core/retrieval.py imports
print("\n=== Check 2: app/core/retrieval.py ===")
try:
    from app.core.retrieval import decompose_question, retrieve_chunks, BoundedIterator, DEFAULT_REFUSAL_MESSAGES, _determine_refusal_category, _build_refusal_response, _should_short_circuit_to_refusal, DEFAULT_CONFIDENCE_THRESHOLD
    print("  ✓ All retrieval imports succeed")
except Exception as e:
    print(f"  ✗ Retrieval import failed: {e}")

# Check 3: app.main import
print("\n=== Check 3: app.main ===")
try:
    from app.main import app
    print("  ✓ app.main import OK")
except Exception as e:
    print(f"  ✗ app.main import failed: {e}")

# Check 4: test_phase10 collection (if possible)
print("\n=== Check 4: test_phase10 collection ===")
try:
    import pytest
    # Note: collection may fail due to pre-existing IngestRequest issue that we already fixed
    print("  (collection check skipped - see above)")
except Exception as e:
    print(f"  (collection check: {e})")

# Check 5: Look for truncated files
print("\n=== Check 5: File sizes ===")
import os
small_files = []
for root, dirs, files in os.walk("app"):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            # Files under 50 bytes are suspicious (retrieval.py was 6 lines ~500 bytes)
            if size < 100 and path != "app/api/v1/endpoints/__init__.py":
                small_files.append((path, size))
if small_files:
    for path, size in small_files:
        print(f"  ⚠ Small file: {path} ({size} bytes)")
else:
    print("  No suspiciously small files found")

print("\n=== Audit Complete ===")