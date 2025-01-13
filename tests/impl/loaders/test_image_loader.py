# tests/impl/loaders/test_image_loader.py
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image, ExifTags
import pytest
from tenacity import RetryError

from src.impl.loaders.image_loader import ImageLoaderConfig, ImageLoader
from src.interfaces.schemas import Document


@pytest.fixture
def tmp_path():
    """Create a custom temporary directory"""
    base_path = Path.cwd()
    temp_dir = Path(base_path / "temp/dir")
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


@pytest.fixture
def sample_image_content():
    """Create a sample image for testing"""
    # Create a new image with a white background
    image = Image.new('RGB', (100, 100), 'white')

    # Save image to bytes buffer
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    return img_buffer.getvalue()


@pytest.fixture
def temp_image_file(tmp_path: Path, sample_image_content):
    """Create a temporary image file for testing"""
    image_path = tmp_path / "test.png"
    image_path.write_bytes(sample_image_content)
    return image_path


@pytest.fixture
def default_config():
    """Create a default image loader configuration"""
    return ImageLoaderConfig(
        name="test_loader",
        type="document_loader",
        supported_formats=["png", "jpg", "jpeg", "tiff", "bmp"],
        chunk_size=500,
        chunk_overlap=20,
        ocr_lang="eng",
        config={}
    )


@pytest.fixture
def image_loader():
    """Create an image loader instance"""
    return ImageLoader()


class TestImageLoader:
    """Test suite for ImageLoader implementation"""

    def test_initialization(self, image_loader, default_config):
        """Test loader initialization with valid config"""
        image_loader.initialize(default_config)
        assert image_loader.config == default_config

    def test_initialization_invalid_config(self, image_loader):
        """Test loader initialization with invalid config"""
        invalid_config = ImageLoaderConfig(
            name="test_loader",
            type="document_loader",
            supported_formats=["doc"],  # No image formats
            chunk_size=100,
            chunk_overlap=20,
            ocr_lang="eng",
            config={}
        )

        with pytest.raises(ValueError):
            image_loader.initialize(invalid_config)

    def test_validate_config(self, image_loader, default_config):
        """Test configuration validation"""
        assert image_loader.validate_config(default_config) == True

        # Test invalid chunk size
        invalid_config = ImageLoaderConfig(
            name="test_loader",
            type="document_loader",
            supported_formats=["png"],
            chunk_size=-1,
            chunk_overlap=20,
            ocr_lang="eng",
            config={}
        )
        assert image_loader.validate_config(invalid_config) == False

        # Test invalid chunk overlap
        invalid_config = ImageLoaderConfig(
            name="test_loader",
            type="document_loader",
            supported_formats=["png"],
            chunk_size=100,
            chunk_overlap=150,  # Larger than chunk size
            ocr_lang="eng",
            config={}
        )
        assert image_loader.validate_config(invalid_config) == False

    def test_supports_format(self, image_loader, default_config):
        """Test format support checking"""
        image_loader.initialize(default_config)
        assert image_loader.supports_format("png") == True
        assert image_loader.supports_format("PNG") == True
        assert image_loader.supports_format("jpg") == True
        assert image_loader.supports_format("doc") == False

    @pytest.mark.asyncio
    async def test_load_without_initialization(self, image_loader, temp_image_file):
        """Test loading without initialization"""
        with pytest.raises(RuntimeError):
            await image_loader.load(temp_image_file)

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, image_loader, default_config):
        """Test loading non-existent file"""
        image_loader.initialize(default_config)
        with pytest.raises(FileNotFoundError):
            await image_loader.load("nonexistent.png")

    @pytest.mark.asyncio
    async def test_load_invalid_format(self, image_loader, default_config, tmp_path):
        """Test loading invalid format"""
        image_loader.initialize(default_config)
        invalid_file = tmp_path / "test.doc"
        invalid_file.touch()

        with pytest.raises(ValueError):
            await image_loader.load(invalid_file)

    @pytest.mark.asyncio
    async def test_load_without_chunking(self, image_loader, temp_image_file):
        """Test loading image without chunking"""
        config = ImageLoaderConfig(
            name="test_loader",
            type="document_loader",
            supported_formats=["png", "jpg", "jpeg", "tiff", "bmp"],
            chunk_size=None,
            chunk_overlap=None,
            ocr_lang="eng",
            config={}
        )
        image_loader.initialize(config)

        # Mock OCR response
        mock_text = "This is a test image with some text content."
        with patch('pytesseract.image_to_string', return_value=mock_text):
            documents = await image_loader.load(temp_image_file)

            assert len(documents) == 1
            assert isinstance(documents[0], Document)
            assert documents[0].content == mock_text
            assert "source" in documents[0].metadata
            assert documents[0].metadata["type"] == "image"
            assert "format" in documents[0].metadata
            assert "size" in documents[0].metadata

    @pytest.mark.asyncio
    async def test_load_with_chunking(self, image_loader, temp_image_file, default_config):
        """Test loading image with chunking"""
        image_loader.initialize(default_config)

        # Mock OCR response with multiple sentences
        mock_text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        with patch('pytesseract.image_to_string', return_value=mock_text):
            documents = await image_loader.load(temp_image_file)

            assert len(documents) >= 1
            for doc in documents:
                assert isinstance(doc, Document)
                assert "chunk_start" in doc.metadata
                assert "chunk_end" in doc.metadata
                assert len(doc.content) <= default_config.chunk_size

    @pytest.mark.asyncio
    async def test_load_with_exif(self, image_loader, tmp_path, default_config):
        """Test loading image with EXIF data"""
        # Create test image with EXIF data
        img = Image.new('RGB', (100, 100), 'white')

        # Create EXIF data using PIL's EXIF tags
        from PIL.ExifTags import TAGS
        exif_data = {
            274: 1,  # Orientation
            271: "Test Manufacturer",  # Make
            272: "Test Model",  # Model
            306: "2024:01:12 00:00:00",  # DateTime
            34855: 400,  # ISOSpeedValue
            37377: (1, 1000),  # ShutterSpeedValue
        }
        # Get the EXIF object
        exif = Image.Exif()

        # Update EXIF data
        for tag_id, value in exif_data.items():
            exif[tag_id] = value
        # Save image with EXIF
        img.save(tmp_path / "test_exif.jpg", format='JPEG', exif=exif.tobytes())
        test_image = tmp_path / "test_exif.jpg"

        # Verify EXIF was written
        with Image.open(test_image) as verify_img:
            assert verify_img._getexif() is not None

        image_loader.initialize(default_config)

        # Mock OCR response
        with patch('pytesseract.image_to_string', return_value="Test content"):
            documents = await image_loader.load(test_image)

            assert len(documents) > 0
            assert "exif" in documents[0].metadata
            # Verify some EXIF data was extracted
            assert documents[0].metadata["exif"]  # Should not be empty
            assert any(key in ['Make', 'Model', 'DateTime']
                       for key in documents[0].metadata["exif"].keys())

    @pytest.mark.asyncio
    async def test_empty_image_text(self, image_loader, temp_image_file, default_config):
        """Test handling of image with no extractable text"""
        image_loader.initialize(default_config)

        # Mock empty OCR response
        with patch('pytesseract.image_to_string', return_value=""):
            documents = await image_loader.load(temp_image_file)

            assert len(documents) == 1
            assert documents[0].content == ""
            assert documents[0].metadata["ocr_status"] == "no_text_found"

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, image_loader, default_config, temp_image_file):
        """Test retry mechanism for image loading"""
        image_loader.initialize(default_config)

        # Mock OCR to fail twice then succeed
        fail_count = 0

        def mock_ocr(*args, **kwargs):
            nonlocal fail_count
            if fail_count < 2:
                fail_count += 1
                raise Exception("Simulated OCR error")
            return "Test content"

        with patch('pytesseract.image_to_string', side_effect=mock_ocr):
            documents = await image_loader.load(temp_image_file)

            # Verify documents were loaded successfully
            assert len(documents) > 0
            # Verify the mock failed twice before succeeding
            assert fail_count == 2

    @pytest.mark.asyncio
    async def test_corrupted_image(self, image_loader, default_config, tmp_path):
        """Test handling of corrupted image file"""
        image_loader.initialize(default_config)

        # Create corrupted image file
        corrupted_file = tmp_path / "corrupted.png"
        corrupted_file.write_text("This is not a valid image file")

        with pytest.raises(RetryError):
            await image_loader.load(corrupted_file)

    @pytest.mark.asyncio
    async def test_different_image_formats(self, image_loader, default_config, tmp_path):
        """Test loading different image formats"""
        image_loader.initialize(default_config)

        # Define format mappings for saving images
        format_mappings = {
            'png': 'PNG',
            'jpg': 'JPEG',
            'bmp': 'BMP'
        }

        # Define expected format names in metadata
        format_expectations = {
            'png': 'png',
            'jpg': ('jpg', 'jpeg'),  # Accept either jpg or jpeg
            'bmp': 'bmp'
        }

        formats = ['png', 'jpg', 'bmp']
        mock_text = "Test content"

        for fmt in formats:
            # Create image in specific format
            img = Image.new('RGB', (100, 100), 'white')
            test_image = tmp_path / f"test.{fmt}"
            img.save(test_image, format=format_mappings[fmt])

            # Test loading
            with patch('pytesseract.image_to_string', return_value=mock_text):
                documents = await image_loader.load(test_image)
                assert len(documents) > 0

                actual_format = documents[0].metadata["format"].lower()
                expected_format = format_expectations[fmt]

                # Handle the case where we accept multiple possible format names
                if isinstance(expected_format, tuple):
                    assert actual_format in expected_format, f"Format {actual_format} not in expected formats {expected_format}"
                else:
                    assert actual_format == expected_format, f"Format {actual_format} != expected format {expected_format}"