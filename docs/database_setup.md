# Database Setup Guide

## ODBC Driver Requirements

A working ODBC driver is required for SQL Server connectivity. The application uses SQL Server ODBC Driver 17 for database operations.

### Installation

For Ubuntu/Debian:
```bash
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql17
```

### Expected Behavior

Without proper ODBC setup:
- Main leads and companies tables may show successful operations
- Properties table operations will fail silently
- Error logs will show: "Can't open lib 'ODBC Driver 17 for SQL Server'"

### Database Schema

The application uses the following tables:

1. leads
   - lead_id (INT, PK)
   - hs_object_id (VARCHAR(50))
   - company_id (INT, FK)
   - email, first_name, last_name

2. companies
   - company_id (INT, PK)
   - hs_object_id (VARCHAR(50))
   - name, city, state

3. lead_properties
   - property_id (INT, PK)
   - lead_id (INT, FK)
   - phone, lifecyclestage
   - competitor_analysis
   - last_response_date

4. company_properties
   - property_id (INT, PK)
   - company_id (INT, FK)
   - annualrevenue
   - xai_facilities_news

### Foreign Key Relationships

- FK_leads_companies: leads.company_id → companies.company_id
- FK_lead_properties: lead_properties.lead_id → leads.lead_id
- FK_company_properties: company_properties.company_id → companies.company_id

### Logging

The application includes comprehensive logging for database operations:
- Debug logs for property operations
- Info logs with IDs and data presence
- Error handling and rollback logging
- Success confirmation logs

Review logs/app.log for detailed database operation tracking.
