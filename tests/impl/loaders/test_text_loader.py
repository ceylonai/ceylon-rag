import io
from pathlib import Path
import pytest

from app.impl.loaders.text_loader import TextLoaderConfig, TextLoader
from app.interfaces.schemas import Document


@pytest.fixture
def tmp_path():
    """Create a custom temporary directory"""
    base_path = Path.cwd()
    temp_dir = Path(base_path / "temp/dir")
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


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
    """Create a default text loader configuration"""
    return TextLoaderConfig(
        name="test_loader",
        type="document_loader",
        chunk_size=200,
        chunk_overlap=20,
        config={}
    )


@pytest.fixture
def text_loader():
    """Create a text loader instance"""
    return TextLoader()


class TestTextLoader:
    """Test suite for TextLoader implementation"""

    def test_initialization(self, text_loader, default_config):
        """Test loader initialization with valid config"""
        text_loader.initialize(default_config)
        assert text_loader.config == default_config

    def test_initialization_invalid_config(self, text_loader):
        """Test loader initialization with invalid config"""
        invalid_config = TextLoaderConfig(
            name="test_loader",
            type="document_loader",
            supported_formats=[],  # Empty formats list
            chunk_size=100,
            chunk_overlap=20,
            config={}
        )

        with pytest.raises(ValueError):
            text_loader.initialize(invalid_config)

    def test_validate_config(self, text_loader, default_config):
        """Test configuration validation"""
        assert text_loader.validate_config(default_config) == True

        # Test invalid chunk size
        invalid_config = TextLoaderConfig(
            name="test_loader",
            type="document_loader",
            chunk_size=-1,
            chunk_overlap=20,
            config={}
        )
        assert text_loader.validate_config(invalid_config) == False

        # Test invalid chunk overlap
        invalid_config = TextLoaderConfig(
            name="test_loader",
            type="document_loader",
            chunk_size=100,
            chunk_overlap=150,  # Larger than chunk size
            config={}
        )
        assert text_loader.validate_config(invalid_config) == False

    def test_supports_format(self, text_loader, default_config):
        """Test format support checking"""
        text_loader.initialize(default_config)
        assert text_loader.supports_format("txt") == True
        assert text_loader.supports_format("TXT") == True
        assert text_loader.supports_format("md") == True
        assert text_loader.supports_format("py") == True
        assert text_loader.supports_format("docx") == False

    @pytest.mark.asyncio
    async def test_load_without_initialization(self, text_loader, temp_text_file):
        """Test loading without initialization"""
        with pytest.raises(RuntimeError):
            await text_loader.load(temp_text_file)

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, text_loader, default_config):
        """Test loading non-existent file"""
        text_loader.initialize(default_config)
        with pytest.raises(FileNotFoundError):
            await text_loader.load("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_load_invalid_format(self, text_loader, default_config, tmp_path):
        """Test loading invalid format"""
        text_loader.initialize(default_config)
        invalid_file = tmp_path / "test.docx"
        invalid_file.touch()

        with pytest.raises(ValueError):
            await text_loader.load(invalid_file)

    @pytest.mark.asyncio
    async def test_load_without_chunking(self, text_loader, temp_text_file):
        """Test loading text without chunking"""
        config = TextLoaderConfig(
            name="test_loader",
            type="document_loader",
            chunk_size=None,
            chunk_overlap=None,
            config={}
        )
        text_loader.initialize(config)

        documents = await text_loader.load(temp_text_file)

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert "source" in documents[0].metadata
        assert "size" in documents[0].metadata
        assert documents[0].metadata["type"] == "text"

    @pytest.mark.asyncio
    async def test_load_with_chunking(self, text_loader, temp_text_file, default_config):
        """Test loading text with chunking"""
        text_loader.initialize(default_config)

        documents = await text_loader.load(temp_text_file)

        assert len(documents) >= 1
        for doc in documents:
            assert isinstance(doc, Document)
            assert "chunk_start" in doc.metadata
            assert "chunk_end" in doc.metadata
            assert len(doc.content) <= default_config.chunk_size

    @pytest.mark.asyncio
    async def test_chunk_overlap(self, text_loader, temp_text_file, default_config):
        """Test chunk overlap functionality"""
        text_loader.initialize(default_config)

        documents = await text_loader.load(temp_text_file)

        # Check adjacent chunks for overlap
        for i in range(len(documents) - 1):
            current_chunk = documents[i].content
            next_chunk = documents[i + 1].content

            # Find overlap between end of current chunk and start of next chunk
            overlap_size = 0
            for j in range(min(len(current_chunk), default_config.chunk_overlap)):
                if current_chunk[-j:] in next_chunk:
                    overlap_size = j
                    break

            assert overlap_size > 0, "No overlap found between adjacent chunks"

    @pytest.mark.asyncio
    async def test_empty_file(self, text_loader, default_config, tmp_path):
        """Test handling of empty file"""
        text_loader.initialize(default_config)

        # Create empty text file
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()

        documents = await text_loader.load(empty_file)
        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_different_text_formats(self, text_loader, default_config, tmp_path, sample_text_content):
        """Test loading different text formats"""
        text_loader.initialize(default_config)

        formats = ['txt', 'md', 'log', 'py', 'json', 'yml']

        for fmt in formats:
            # Create file in specific format
            test_file = tmp_path / f"test.{fmt}"
            test_file.write_text(sample_text_content)

            # Test loading
            documents = await text_loader.load(test_file)
            assert len(documents) > 0
            assert documents[0].metadata["type"] == "text"

    @pytest.mark.asyncio
    async def test_unicode_content(self, text_loader, default_config, tmp_path):
        """Test loading file with Unicode content"""
        text_loader.initialize(default_config)

        # Create text file with Unicode content
        unicode_content = """Hello World! 
        Γειά σου Κόσμε! 
        こんにちは世界！
        안녕하세요 세상!
        مرحبا بالعالم!
        """
        unicode_file = tmp_path / "unicode.txt"
        unicode_file.write_text(unicode_content, encoding='utf-8')

        documents = await text_loader.load(unicode_file)
        assert len(documents) > 0
        assert all(line in documents[0].content for line in unicode_content.splitlines())

    @pytest.mark.asyncio
    async def test_large_file(self, text_loader, default_config, tmp_path):
        """Test loading and chunking of large file"""
        text_loader.initialize(default_config)

        # Create large text file
        large_content = "This is a test sentence.\n" * 1000
        large_file = tmp_path / "large.txt"
        large_file.write_text(large_content)

        documents = await text_loader.load(large_file)
        assert len(documents) > 1
        print(len(documents[0].content))
        assert all(len(doc.content) <= default_config.chunk_size for doc in documents)


    @pytest.mark.asyncio
    async def test_whitespace_handling(self, text_loader, default_config, tmp_path):
        """Test handling of various whitespace patterns"""
        text_loader.initialize(default_config)

        # Create text with various whitespace patterns
        whitespace_content = """

        First line with content.
            Indented line.

        Multiple empty lines above.
        Trailing spaces:    
            Mixed indentation.

        """
        whitespace_file = tmp_path / "whitespace.txt"
        whitespace_file.write_text(whitespace_content)

        documents = await text_loader.load(whitespace_file)
        assert len(documents) > 0
        # Check that excessive whitespace is normalized
        assert not documents[0].content.startswith('\n\n')
        assert not documents[0].content.endswith('\n\n')


if __name__ == "__main__":
    pytest.main([__file__])