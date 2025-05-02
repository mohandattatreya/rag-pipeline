from typing import List
import logging

logger = logging.getLogger(__name__)

def chunk_text_by_char(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """Splits text into chunks based on character count with overlap."""
    if chunk_overlap >= chunk_size:
        logger.warning(f"Chunk overlap ({chunk_overlap}) is >= chunk size ({chunk_size}). Setting overlap to 0.")
        chunk_overlap = 0

    chunks = []
    start_index = 0
    text_len = len(text)

    while start_index < text_len:
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        chunks.append(chunk)

        # Move start index for the next chunk
        next_start_index = start_index + chunk_size - chunk_overlap

        # If the next start index is the same as the current one due to zero step,
        # advance by chunk_size to prevent infinite loop.
        if next_start_index <= start_index:
            if start_index + chunk_size >= text_len:
                 # Already processed the last part
                 break
            else:
                # Force move forward if step is zero and not at the end
                logger.debug("Chunk step size is zero or negative, forcing advance by chunk_size")
                next_start_index = start_index + chunk_size

        start_index = next_start_index

    logger.info(f"Split text of length {text_len} into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_text = "This is a sample text designed to be split into several overlapping chunks for demonstration purposes. It needs to be long enough."
    chunk_size = 30
    chunk_overlap = 10

    print(f"Original text (length {len(test_text)}): {test_text}")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    print("---")

    chunks = chunk_text_by_char(test_text, chunk_size, chunk_overlap)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: (length {len(chunk)})\n'{chunk}'")
        # Verify overlap with the next chunk if it exists
        if i < len(chunks) - 1:
            overlap_actual = chunks[i+1][:chunk_overlap]
            expected_overlap_area = chunk[-chunk_overlap:]
            print(f"  Expected overlap area: '{expected_overlap_area}'")
            print(f"  Actual next chunk start: '{overlap_actual}'")
            if overlap_actual != expected_overlap_area:
                 print(f"  Overlap mismatch detected!", flush=True)
        print("---")

    # Test edge case: overlap >= size
    print("\nTesting overlap >= size:")
    chunks_edge = chunk_text_by_char(test_text, chunk_size=20, chunk_overlap=25)
    # Should log a warning and behave as if overlap is 0
    print(f"Generated {len(chunks_edge)} chunks for overlap >= size test.")

    # Test edge case: text shorter than chunk size
    print("\nTesting text shorter than chunk size:")
    short_text = "Short text."
    chunks_short = chunk_text_by_char(short_text, chunk_size=50, chunk_overlap=10)
    print(f"Generated {len(chunks_short)} chunk(s) for short text: {chunks_short}") 