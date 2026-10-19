from app.core.chunking import chunk_text

def test_chunking_normal():
    text = "This is a test. " * 10  # 20 words, each word is 5 letters plus space and period? Let's just use a string of known length.
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    # We expect the length of chunks to be such that we cover the text with overlap.
    # Let's just check that we don't get an error and that the chunks are non-empty.
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= 10  # chunk size is 10
        # Because of overlap, the chunk might be less than 10 at the end? Actually, we take exactly chunk_size characters unless we are at the end.
        # Our function takes exactly chunk_size characters from start to start+chunk_size, so each chunk is exactly chunk_size except possibly the last?
        # Actually, we take text[start:end] where end = start+chunk_size, so if the text is shorter, we get less.
        # So we can only assert that the chunk length is <= chunk_size.
        assert len(chunk) <= 10

def test_chunking_overlap_gte_chunk_size():
    try:
        chunk_text("hello", chunk_size=5, overlap=5)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Overlap" in str(e) and "must be less than chunk size" in str(e)

def test_chunking_empty_text():
    try:
        chunk_text("", chunk_size=10, overlap=2)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Text must not be empty or whitespace only" in str(e)

def test_chunking_whitespace_only_text():
    try:
        chunk_text("   \n\t  ", chunk_size=10, overlap=2)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Text must not be empty or whitespace only" in str(e)

if __name__ == "__main__":
    test_chunking_normal()
    test_chunking_overlap_gte_chunk_size()
    test_chunking_empty_text()
    test_chunking_whitespace_only_text()
    print("All chunking tests passed!")