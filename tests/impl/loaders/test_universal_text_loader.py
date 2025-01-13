import pytest
from pathlib import Path
from typing import List

from app.impl.loaders.universal_text_loader import UniversalTextLoader, LoaderConfig, Document


@pytest.fixture
def tmp_path():
    """Create a temporary directory for test files"""
    base_path = Path.cwd()
    temp_dir = Path(base_path / "temp/test_dir")
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    # Cleanup after tests
    for file in temp_dir.glob("*"):
        file.unlink()
    temp_dir.rmdir()


@pytest.fixture
def sample_text_content():
    """Create sample text content for testing"""
    return """This is a test document.
    It contains multiple lines of text.
    Some lines are short.
    Other lines are much longer and contain multiple sentences. These sentences help test the chunking functionality.
    Here's another paragraph with different content.

    And a final paragraph with more text to ensure we have enough content for testing various scenarios.
    This includes checking how the loader handles whitespace, line breaks, and different sentence structures."""


@pytest.fixture
def temp_text_file(tmp_path: Path, sample_text_content):
    """Create a temporary text file for testing"""
    text_path = tmp_path / "test.txt"
    text_path.write_text(sample_text_content, encoding='utf-8')
    return text_path


@pytest.fixture
def default_config():
    """Create a default loader configuration"""
    return LoaderConfig(
        chunk_size=200,
        chunk_overlap=20,
        encoding_fallbacks=['utf-8', 'latin-1', 'ascii']
    )


@pytest.fixture
def loader():
    """Create a loader instance"""
    return UniversalTextLoader()


class TestUniversalTextLoader:
    """Test suite for UniversalTextLoader implementation"""

    def test_initialization(self, loader, default_config):
        """Test loader initialization with different configs"""
        # Test with default config
        loader_with_config = UniversalTextLoader(default_config)
        assert loader_with_config.config == default_config

        # Test without config (should use default values)
        default_loader = UniversalTextLoader()
        assert default_loader.config is not None
        assert default_loader.config.chunk_size is None
        assert default_loader.config.chunk_overlap is None

    def test_load_nonexistent_file(self, loader):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.txt")

    def test_load_without_chunking(self, loader, temp_text_file, sample_text_content):
        """Test loading text without chunking"""
        documents = loader.load(temp_text_file)

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert documents[0].content.strip() == sample_text_content.strip()
        assert "source" in documents[0].metadata
        assert "size" in documents[0].metadata

    def test_load_with_chunking(self, default_config, temp_text_file):
        """Test loading text with chunking"""
        loader = UniversalTextLoader(default_config)
        documents = loader.load(temp_text_file)

        assert len(documents) > 1
        for doc in documents:
            assert isinstance(doc, Document)
            assert "chunk_start" in doc.metadata
            assert "chunk_end" in doc.metadata
            assert len(doc.content) <= default_config.chunk_size

    def test_chunk_overlap(self, default_config, temp_text_file):
        """Test chunk overlap functionality"""
        loader = UniversalTextLoader(default_config)
        documents = loader.load(temp_text_file)

        # Check adjacent chunks for overlap
        for i in range(len(documents) - 1):
            current_chunk = documents[i].content
            next_chunk = documents[i + 1].content

            # Find overlap between end of current chunk and start of next chunk
            overlap_found = False
            min_overlap = min(len(current_chunk), default_config.chunk_overlap)
            for j in range(min_overlap, 0, -1):
                if current_chunk[-j:] in next_chunk:
                    overlap_found = True
                    break

            assert overlap_found, "No overlap found between adjacent chunks"

    def test_empty_file(self, loader, tmp_path):
        """Test handling of empty file"""
        empty_file = tmp_path / "empty.xyz"
        empty_file.touch()

        documents = loader.load(empty_file)
        assert len(documents) == 0

    def test_binary_file(self, loader, tmp_path):
        """Test loading binary file"""
        # Create a binary file
        binary_file = tmp_path / "binary.bin"
        with open(binary_file, 'wb') as f:
            f.write(bytes(range(256)))

        documents = loader.load(binary_file)
        assert len(documents) > 0
        assert isinstance(documents[0].content, str)

    def test_unicode_content(self, loader, tmp_path):
        """Test loading file with Unicode content"""
        unicode_content = """Hello World! 
        Γειά σου Κόσμε! 
        こんにちは世界！
        안녕하세요 세상!
        مرحبا بالعالم!
        """
        unicode_file = tmp_path / "unicode.xyz"
        unicode_file.write_text(unicode_content, encoding='utf-8')

        documents = loader.load(unicode_file)
        assert len(documents) > 0
        assert all(line in documents[0].content for line in unicode_content.splitlines())

    def test_different_encodings(self, loader, tmp_path):
        """Test loading files with different encodings"""
        # Test UTF-8 with BOM
        content = "Hello, UTF-8 with BOM!"
        file_path = tmp_path / "utf8_bom.txt"
        with open(file_path, 'wb') as f:
            f.write(b'\xef\xbb\xbf' + content.encode('utf-8'))

        documents = loader.load(file_path)
        assert len(documents) > 0
        assert documents[0].content.strip() == content  # Should match exactly as BOM is stripped

        # Test Latin-1 content
        latin1_content = "Hello, Latin-1 £ symbol!"
        latin1_file = tmp_path / "latin1.txt"
        with open(latin1_file, 'wb') as f:
            f.write(latin1_content.encode('latin-1'))


        documents = loader.load(latin1_file)
        assert len(documents) > 0
        assert documents[0].content.strip() == latin1_content

    def test_large_file(self, default_config, tmp_path):
        """Test loading and chunking of large file"""
        loader = UniversalTextLoader(default_config)

        # Create large text file
        large_content = "This is a test sentence.\n" * 1000
        large_file = tmp_path / "large.xyz"
        large_file.write_text(large_content)

        documents = loader.load(large_file)
        assert len(documents) > 1
        assert all(len(doc.content) <= default_config.chunk_size for doc in documents)

    def test_whitespace_handling(self, loader, tmp_path):
        """Test handling of various whitespace patterns"""
        whitespace_content = """

        First line with content.
            Indented line.

        Multiple empty lines above.
        Trailing spaces:    
            Mixed indentation.

        """
        whitespace_file = tmp_path / "whitespace.xyz"
        whitespace_file.write_text(whitespace_content)

        documents = loader.load(whitespace_file)
        assert len(documents) > 0
        # Check that content is preserved but excessive whitespace is normalized
        assert documents[0].content.strip() == whitespace_content.strip()
        assert not documents[0].content.startswith('\n\n')
        assert not documents[0].content.endswith('\n\n')

    def test_mixed_line_endings(self, loader, tmp_path):
        """Test handling of mixed line endings"""
        content = "Line 1\rLine 2\r\nLine 3\nLine 4"
        mixed_file = tmp_path / "mixed_endings.xyz"
        mixed_file.write_bytes(content.encode('utf-8'))

        documents = loader.load(mixed_file)
        assert len(documents) > 0
        # Check that all line endings are normalized
        assert '\r\n' not in documents[0].content
        assert '\r' not in documents[0].content
        assert len(documents[0].content.split('\n')) == 4


if __name__ == "__main__":
    pytest.main([__file__])