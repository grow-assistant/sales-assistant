# Sales Assistant

A Python-based automated sales assistant designed for golf clubs, leveraging AI for personalized outreach and lead management.

## Overview

The Sales Assistant automates and enhances the sales outreach process for golf clubs by:
- Fetching and qualifying leads from HubSpot
- Generating personalized email content using AI (OpenAI GPT-4 and xAI/Grok)
- Managing follow-up sequences
- Integrating with Gmail for email drafts and sending
- Tracking interactions and responses in a SQL database

## Project Structure

```
sales-assistant/
├── agents/                 # Core AI agent logic
│   └── orchestrator.py     # Main workflow coordination
├── hubspot_integration/    # HubSpot CRM integration
├── scheduling/            # Follow-up and database operations
│   ├── followup_generation.py
│   └── sql_lookup.py
├── utils/                # Utility functions and integrations
│   ├── gmail_integration.py
│   ├── xai_integration.py
│   └── logging_setup.py
├── external/             # External API integrations
├── config/              # Configuration and settings
└── docs/templates/      # Email templates and decision trees
```

## Prerequisites

- Python 3.10
- Conda package manager
- Access to required services:
  - HubSpot CRM account
  - OpenAI API access
  - xAI API access
  - Gmail API credentials
  - SQL database

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/grow-assistant/sales-assistant.git
cd sales-assistant
```

2. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate sales-assistant  # The environment name from environment.yml
```

3. Configure environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` file with your API keys and credentials:
```ini
# Required API Keys
OPENAI_API_KEY=your_openai_api_key
HUBSPOT_API_KEY=your_hubspot_api_key
XAI_TOKEN=your_xai_token

# Optional Configuration
XAI_API_URL=https://api.x.ai/v1/chat/completions
XAI_MODEL=grok-beta
DEV_MODE=false
DEBUG_MODE=false
```

Note: The project includes both `environment.yml` (Conda) and `requirements.txt`. We recommend using Conda with `environment.yml` for the most consistent environment setup.

## Gmail Credentials Setup

1. Go to Google Cloud Console and enable the Gmail API for your project.
2. Create an OAuth 2.0 Client ID (Desktop application type).
3. Download the JSON credentials file and rename it to `credentials.json`.
4. Place `credentials.json` in the project root directory (it is already in .gitignore).
5. On first run, a browser window will open to authenticate. Once completed, a `token.json` file is created locally.
6. Keep these files private and do not commit them to the repository.

Note: The application requires the `https://www.googleapis.com/auth/gmail.modify` scope for creating and managing email drafts. If you modify the scopes, delete `token.json` to force re-authentication.

## Key Components

### 1. Agent System (`/agents`)
- Orchestrates the entire sales workflow
- Manages lead context preparation
- Handles email personalization using AI
- Coordinates between different services

### 2. HubSpot Integration (`/hubspot_integration`)
- Fetches lead data from HubSpot CRM
- Manages contact information
- Handles data enrichment
- Maintains CRM synchronization

### 3. Scheduling System (`/scheduling`)
- Generates follow-up emails
- Manages database operations
- Tracks lead interactions
- Handles SQL operations for lead data

### 4. Utility Functions (`/utils`)
- Gmail integration for email management
- xAI integration for advanced AI features
- Logging setup and configuration
- Document reading and processing

## Usage Examples

### Basic Lead Processing
```python
from agents.orchestrator import process_lead
from scheduling.sql_lookup import build_lead_sheet_from_sql

# Process a lead by email
lead_email = "example@golfclub.com"
lead_sheet = build_lead_sheet_from_sql(lead_email)
result = process_lead(lead_sheet)
```

### Email Generation
```python
from utils.xai_integration import personalize_email_with_xai

# Generate personalized email
subject = "Golf Club Operations Enhancement"
body = "Template email body..."
new_subject, new_body = personalize_email_with_xai(lead_sheet, subject, body)
```

### Gmail Integration
```python
from utils.gmail_integration import create_draft

# Create email draft
result = create_draft(
    sender="your-email@example.com",
    to="lead@golfclub.com",
    subject=new_subject,
    message_text=new_body
)
```

## Development Notes

- Set `DEBUG_MODE=true` in `.env` for detailed logging
- Use `DEV_MODE=true` for testing without sending actual emails
- Check `docs/templates/` for email templates and decision trees
- The project uses SQLite for local development; configure your production database in `.env`

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Submit a pull request with a clear description of changes

## License

Proprietary - All rights reserved
