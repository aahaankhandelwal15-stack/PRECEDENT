import sys
with open('D:/Projects/PRECEDENT/app/api/v1/endpoints/ingest.py', 'r') as f:
    lines = f.readlines()
# indices are 0-indexed
# we want to replace lines 85 to 107 (inclusive) with new block
new_block = '''    # 7. Create chunk records
    for index, chunk in enumerate(chunks):
        chunk_id = f\"{document_id}_{index}\"
        new_chunk = PolicyChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=chunk,
            policy_id=request.policy_id,
            version=request.version,
            effective_from=request.effective_from,
            effective_until=request.effective_until,
            owner=request.owner,
            authority_level=request.authority_level,
            jurisdiction=request.jurisdiction,
            department=request.department,
            audience=request.audience,
            status=request.status,
            supersedes=request.supersedes,  # This is the policy_id that this chunk supersedes (if any)
            # Note: The model has a `supersedes` field on PolicyChunk, which is the policy_id of the document this chunk's document supersedes.
            # We are setting it to the request.supersedes (which is a policy_id).
            superseded_by=None  # Will be set if another document supersedes this one
        )
        session.add(new_chunk)
'''
# Make sure new_block ends with a newline? We'll split and then add newline.
new_block_lines = new_block.splitlines(keepends=True)
# Now, replace
lines[85:108] = new_block_lines
with open('D:/Projects/PRECEDENT/app/api/v1/endpoints/ingest.py', 'w') as f:
    f.writelines(lines)