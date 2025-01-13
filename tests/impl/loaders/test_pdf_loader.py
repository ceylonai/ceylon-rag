import io
from pathlib import Path
from unittest.mock import patch

import pytest
from pypdf import PdfWriter, PageObject, PdfReader
from pypdf.errors import PdfReadError, PageSizeNotDefinedError
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.impl.loaders.pdf_loader import PDFLoaderConfig, PDFLoader
from src.interfaces.schemas import Document


@pytest.fixture
def tmp_path():
    """Create a custom temporary directory"""
    base_path = Path.cwd()  # Run location path
    temp_dir = Path(base_path / "temp/dir")  # Change this to your desired path
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


@pytest.fixture
def sample_pdf_content():
    # Create PDF buffer
    pdf_buffer = io.BytesIO()

    # Create a canvas to write text
    can = canvas.Canvas(pdf_buffer, pagesize=letter)

    # Create 3 pages with sample content
    for i in range(3):
        can.drawString(72, 720, f"This is page {i + 1} content.")
        can.drawString(72, 700, "It contains multiple sentences.")
        can.drawString(72, 680, "Each page has unique text for testing.")
        can.showPage()

    can.save()
    pdf_buffer.seek(0)

    # Create final PDF with all pages
    final_pdf = io.BytesIO()
    pdf_writer = PdfWriter()

    # Add all pages from the buffer
    pdf_reader = PdfReader(pdf_buffer)
    for page in pdf_reader.pages:
        pdf_writer.add_page(page)

    # Write the final PDF
    pdf_writer.write(final_pdf)
    final_pdf.seek(0)

    return final_pdf.getvalue()


@pytest.fixture
def temp_pdf_file(tmp_path: Path, sample_pdf_content):
    """Create a temporary PDF file for testing"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(sample_pdf_content)
    return pdf_path


@pytest.fixture
def default_config():
    """Create a default PDF loader configuration"""
    return PDFLoaderConfig(
        name="test_loader",
        type="document_loader",
        supported_formats=["pdf"],
        chunk_size=500,
        chunk_overlap=20,
        config={}
    )


@pytest.fixture
def pdf_loader():
    """Create a PDF loader instance"""
    return PDFLoader()


class TestPDFLoader:
    """Test suite for PDFLoader implementation"""

    def test_initialization(self, pdf_loader, default_config):
        """Test loader initialization with valid config"""
        pdf_loader.initialize(default_config)
        assert pdf_loader.config == default_config

    def test_initialization_invalid_config(self, pdf_loader):
        """Test loader initialization with invalid config"""
        invalid_config = PDFLoaderConfig(
            name="test_loader",
            type="document_loader",
            supported_formats=["doc"],  # PDF not supported
            chunk_size=100,
            chunk_overlap=20,
            config={}
        )

        with pytest.raises(ValueError):
            pdf_loader.initialize(invalid_config)

    def test_validate_config(self, pdf_loader, default_config):
        """Test configuration validation"""
        assert pdf_loader.validate_config(default_config) == True

        # Test invalid chunk size
        invalid_config = PDFLoaderConfig(
            name="test_loader",
            type="document_loader",
            supported_formats=["pdf"],
            chunk_size=-1,
            chunk_overlap=20,
            config={}
        )
        assert pdf_loader.validate_config(invalid_config) == False

        # Test invalid chunk overlap
        invalid_config = PDFLoaderConfig(
            name="test_loader",
            type="document_loader",
            supported_formats=["pdf"],
            chunk_size=100,
            chunk_overlap=150,  # Larger than chunk size
            config={}
        )
        assert pdf_loader.validate_config(invalid_config) == False

    def test_supports_format(self, pdf_loader, default_config):
        """Test format support checking"""
        pdf_loader.initialize(default_config)
        assert pdf_loader.supports_format("pdf") == True
        assert pdf_loader.supports_format("PDF") == True
        assert pdf_loader.supports_format("doc") == False

    @pytest.mark.asyncio
    async def test_load_without_initialization(self, pdf_loader, temp_pdf_file):
        """Test loading without initialization"""
        with pytest.raises(RuntimeError):
            await pdf_loader.load(temp_pdf_file)

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, pdf_loader, default_config):
        """Test loading non-existent file"""
        pdf_loader.initialize(default_config)
        with pytest.raises(FileNotFoundError):
            await pdf_loader.load("nonexistent.pdf")

    @pytest.mark.asyncio
    async def test_load_invalid_format(self, pdf_loader, default_config, tmp_path):
        """Test loading invalid format"""
        pdf_loader.initialize(default_config)
        invalid_file = tmp_path / "test.doc"
        invalid_file.touch()

        with pytest.raises(ValueError):
            await pdf_loader.load(invalid_file)

    @pytest.mark.asyncio
    async def test_load_without_chunking(self, pdf_loader, temp_pdf_file):
        """Test loading PDF without chunking"""
        config = PDFLoaderConfig(
            name="test_loader",
            type="document_loader",
            supported_formats=["pdf"],
            chunk_size=None,
            chunk_overlap=None,
            config = {}
        )
        pdf_loader.initialize(config)

        documents = await pdf_loader.load(temp_pdf_file)

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert "source" in documents[0].metadata
        assert "pages" in documents[0].metadata
        assert documents[0].metadata["type"] == "pdf"

    @pytest.mark.asyncio
    async def test_load_with_chunking(self, pdf_loader, temp_pdf_file, default_config):
        """Test loading PDF with chunking"""
        pdf_loader.initialize(default_config)

        documents = await pdf_loader.load(temp_pdf_file)

        assert len(documents) >= 1
        for doc in documents:
            assert isinstance(doc, Document)
            assert "chunk_start" in doc.metadata
            assert "chunk_end" in doc.metadata
            assert len(doc.content) <= default_config.chunk_size

    @pytest.mark.asyncio
    async def test_chunk_overlap(self, pdf_loader, temp_pdf_file, default_config):
        """Test chunk overlap functionality"""
        pdf_loader.initialize(default_config)

        documents = await pdf_loader.load(temp_pdf_file)

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
    async def test_retry_mechanism(self, pdf_loader, default_config, tmp_path, sample_pdf_content):
        """Test retry mechanism for PDF loading"""
        pdf_loader.initialize(default_config)

        # Create a test PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(sample_pdf_content)

        # Mock PdfReader to fail twice then succeed
        fail_count = 0
        original_reader = PdfReader

        class MockPdfReader:
            def __init__(self, *args, **kwargs):
                nonlocal fail_count
                if fail_count < 2:
                    fail_count += 1
                    raise PageSizeNotDefinedError("Simulated PDF read error")
                self.real_reader = original_reader(*args, **kwargs)

            @property
            def pages(self):
                return self.real_reader.pages

        # Use the mock PdfReader with the correct import path
        with patch('src.impl.loaders.pdf_loader.PdfReader', MockPdfReader):
            documents = await pdf_loader.load(test_pdf)

            # Verify documents were loaded successfully
            assert len(documents) > 0
            # Verify the mock failed twice before succeeding
            assert fail_count == 2
    @pytest.mark.asyncio
    async def test_empty_pdf(self, pdf_loader, default_config, tmp_path):
        """Test handling of empty PDF"""
        pdf_loader.initialize(default_config)

        # Create empty PDF
        pdf_path = tmp_path / "empty.pdf"
        pdf_writer = PdfWriter()
        with open(pdf_path, 'wb') as f:
            pdf_writer.write(f)

        documents = await pdf_loader.load(pdf_path)
        assert len(documents) == 0


if __name__ == "__main__":
    pytest.main([__file__])