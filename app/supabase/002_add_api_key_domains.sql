-- Migration: Add allowed_domains to api_keys table
-- This allows restricting API key usage to specific domains

-- Add allowed_domains column to api_keys table
ALTER TABLE api_keys 
ADD COLUMN IF NOT EXISTS allowed_domains TEXT[] DEFAULT NULL;

-- Comment explaining the field
COMMENT ON COLUMN api_keys.allowed_domains IS 
'Array of allowed domains for this API key (e.g., ["example.com", "www.example.com"]). 
NULL or empty array means no restrictions (allow all domains).
Wildcards supported: *.example.com matches any subdomain.';

-- Create index for faster lookups (if we query by domain in future)
-- Note: PostgreSQL doesn't index arrays directly, but we can use GIN index if needed later
-- For now, we fetch the record and check domains in application code

-- Update existing rows to have empty array (no restriction by default)
UPDATE api_keys 
SET allowed_domains = NULL 
WHERE allowed_domains IS NULL;
