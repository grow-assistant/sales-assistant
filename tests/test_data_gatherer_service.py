import pytest
import json
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime

from services.data_gatherer_service import DataGathererService
from utils.logging_setup import logger

@pytest.fixture
def data_gatherer():
    """Create a DataGathererService instance with mocked dependencies."""
    with patch('services.data_gatherer_service.HubspotService') as mock_hubspot:
        service = DataGathererService()
        service._hubspot = mock_hubspot
        return service

@pytest.fixture
def mock_lead_data():
    """Sample lead data for testing."""
    return {
        "id": "12345",
        "properties": {
            "email": "test@example.com",
            "firstname": "Test",
            "lastname": "User",
            "phone": "123-456-7890"
        },
        "emails": ["Email 1", "Email 2"]
    }

@pytest.fixture
def mock_company_data():
    """Sample company data for testing."""
    return {
        "id": "67890",
        "properties": {
            "name": "Test Company",
            "city": "Test City",
            "state": "TS",
            "domain": "testcompany.com",
            "website": "https://testcompany.com"
        }
    }

class TestDataGathererService:
    """Test suite for DataGathererService."""

    def test_gather_lead_data_success(self, data_gatherer, mock_lead_data, mock_company_data):
        """Test successful lead data gathering with all components."""
        # Setup mock responses for individual HubSpot API calls
        data_gatherer._hubspot.get_contact_by_email.return_value = {
            "id": mock_lead_data["id"],
            "properties": mock_lead_data["properties"]
        }
        data_gatherer._hubspot.get_contact_properties.return_value = mock_lead_data["properties"]
        data_gatherer._hubspot.get_associated_company_id.return_value = mock_company_data["id"]
        data_gatherer._hubspot.get_company_data.return_value = mock_company_data["properties"]

        # Create Mock objects for the analysis methods
        mock_competitor = Mock(return_value={})
        mock_research = Mock(return_value={})
        mock_interactions = Mock(return_value=[])
        mock_season = Mock(return_value={})

        # Replace the instance methods with Mock objects
        data_gatherer.check_competitor_on_website = mock_competitor
        data_gatherer.market_research = mock_research
        data_gatherer.review_previous_interactions = mock_interactions
        data_gatherer.determine_club_season = mock_season

        # Execute
        result = data_gatherer.gather_lead_data("test@example.com")

        # Verify structure
        assert "metadata" in result
        assert "lead_data" in result
        assert "analysis" in result

        # Verify metadata
        assert result["metadata"]["status"] == "success"
        assert result["metadata"]["contact_id"] == mock_lead_data["id"]
        assert result["metadata"]["company_id"] == mock_company_data["id"]

        # Verify lead data
        assert result["lead_data"]["id"] == mock_lead_data["id"]
        assert result["lead_data"]["properties"] == mock_lead_data["properties"]
        assert "company_data" in result["lead_data"]
        assert result["lead_data"]["company_data"]["city"] == mock_company_data["properties"]["city"]
        assert result["lead_data"]["company_data"]["state"] == mock_company_data["properties"]["state"]

    def test_gather_lead_data_no_company(self, data_gatherer, mock_lead_data):
        """Test lead data gathering without company data."""
        # Setup mock responses for individual HubSpot API calls
        data_gatherer._hubspot.get_contact_by_email.return_value = {
            "id": mock_lead_data["id"],
            "properties": mock_lead_data["properties"]
        }
        data_gatherer._hubspot.get_contact_properties.return_value = mock_lead_data["properties"]
        data_gatherer._hubspot.get_associated_company_id.return_value = None
        data_gatherer._hubspot.get_company_data.return_value = {}

        # Create Mock objects for the analysis methods
        mock_competitor = Mock(return_value={})
        mock_research = Mock(return_value={})
        mock_interactions = Mock(return_value=[])
        mock_season = Mock(return_value={})

        # Replace the instance methods with Mock objects
        data_gatherer.check_competitor_on_website = mock_competitor
        data_gatherer.market_research = mock_research
        data_gatherer.review_previous_interactions = mock_interactions
        data_gatherer.determine_club_season = mock_season

        # Execute
        result = data_gatherer.gather_lead_data("test@example.com")

        # Verify structure
        assert "metadata" in result
        assert "lead_data" in result
        assert "analysis" in result

        # Verify empty company data handling
        assert result["lead_data"]["company_data"] == {
            "hs_object_id": "",
            "name": "",
            "city": "",
            "state": "",
            "domain": "",
            "website": ""
        }

    def test_determine_club_season_success(self, data_gatherer):
        """Test successful season determination."""
        result = data_gatherer.determine_club_season("Test City", "FL")

        # Verify response format
        assert "year_round" in result
        assert "start_month" in result
        assert "end_month" in result
        assert "peak_season_start" in result
        assert "peak_season_end" in result
        assert "status" in result
        assert "error" in result

        # Verify status
        assert result["status"] in ["success", "no_data"]

    def test_determine_club_season_no_data(self, data_gatherer):
        """Test season determination with no location data."""
        result = data_gatherer.determine_club_season("", "")

        assert result["status"] == "no_data"
        assert result["error"] == "No location data provided"
        assert result["year_round"] == "Unknown"
        assert result["peak_season_start"] == "05-01"
        assert result["peak_season_end"] == "08-31"

    def test_save_lead_context_preserves_sensitive_data(self, data_gatherer, mock_lead_data, tmp_path):
        """Test that sensitive data is preserved when saving lead context."""
        # Setup test data
        test_email = "sensitive@example.com"
        test_phone = "123-456-7890"
        lead_sheet = {
            "metadata": {"contact_id": "12345", "status": "success"},
            "lead_data": {
                "properties": {
                    "email": test_email,
                    "phone": test_phone
                },
                "emails": ["Email content 1", "Email content 2"]
            }
        }

        # Patch the context directory to use tmp_path
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.open', create=True), \
             patch('json.dump') as mock_json_dump:

            # Execute
            data_gatherer._save_lead_context(lead_sheet, test_email)

            # Get the saved data that was passed to json.dump
            saved_data = mock_json_dump.call_args[0][0]

            # Verify data is preserved unmasked
            assert saved_data["lead_data"]["properties"]["email"] == test_email
            assert saved_data["lead_data"]["properties"]["phone"] == test_phone
            assert saved_data["lead_data"]["emails"] == ["Email content 1", "Email content 2"]

    def test_month_conversion_methods(self, data_gatherer):
        """Test month name to date conversion methods."""
        # Test valid month
        assert data_gatherer._month_to_first_day("January") == "01-01"
        assert data_gatherer._month_to_last_day("January") == "01-31"

        # Test invalid month
        assert data_gatherer._month_to_first_day("InvalidMonth") == "05-01"
        assert data_gatherer._month_to_last_day("InvalidMonth") == "08-31"

        # Test edge cases
        assert data_gatherer._month_to_first_day("February") == "02-01"
        assert data_gatherer._month_to_last_day("February") == "02-28"
