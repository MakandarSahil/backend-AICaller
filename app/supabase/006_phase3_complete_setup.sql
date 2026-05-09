-- =============================================================================
-- PHASE 3 COMPLETE SETUP: BYO Twilio Production Flow
-- Date: 2026-05-09
-- Description: All database changes needed for Phase 3
-- Run this entire file in Supabase SQL Editor
-- =============================================================================

-- =============================================================================
-- 1. PROVIDER SECRETS TABLE (Encrypted credential storage)
-- =============================================================================
-- This table stores encrypted Twilio credentials
-- Encryption is done by the backend using Fernet (Python cryptography library)

CREATE TABLE IF NOT EXISTS provider_secrets (
    id TEXT PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    encrypted_data TEXT NOT NULL,  -- Fernet-encrypted JSON: {"account_sid": "...", "auth_token": "..."}
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE provider_secrets ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Only workspace members can access their own secrets
CREATE POLICY "provider_secrets_workspace_isolation"
    ON provider_secrets FOR ALL
    USING (workspace_id = my_workspace_id())
    WITH CHECK (workspace_id = my_workspace_id());

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_provider_secrets_workspace 
    ON provider_secrets(workspace_id);

COMMENT ON TABLE provider_secrets IS 
    'Encrypted storage for telephony provider credentials (Twilio, etc.).
     Credentials are encrypted by backend using Fernet before storage.
     The ENCRYPTION_KEY is stored only in backend environment variables.
     Never stores plaintext credentials.';

-- =============================================================================
-- 2. VERIFY WORKSPACE_TELEPHONY_PROVIDERS TABLE EXISTS
-- =============================================================================
-- This should already exist from V4 schema, but verify/creating just in case

CREATE TABLE IF NOT EXISTS workspace_telephony_providers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,  -- 'twilio', 'vonage', 'plivo', etc.
    display_name TEXT NOT NULL,
    provider_type TEXT NOT NULL DEFAULT 'own',  -- 'platform' or 'own'
    vault_secret_id TEXT,  -- References provider_secrets.id (our encrypted storage)
    is_active BOOLEAN NOT NULL DEFAULT true,
    is_verified BOOLEAN NOT NULL DEFAULT false,
    verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- One provider type per workspace (can be relaxed later for multiple Twilio accounts)
    CONSTRAINT provider_unique_per_workspace UNIQUE (workspace_id, provider, provider_type)
);

-- Enable RLS if not already enabled
ALTER TABLE workspace_telephony_providers ENABLE ROW LEVEL SECURITY;

-- Drop existing policy if exists to avoid conflicts
DROP POLICY IF EXISTS "telephony_providers_owner" ON workspace_telephony_providers;

-- Create RLS policy
CREATE POLICY "telephony_providers_owner"
    ON workspace_telephony_providers FOR ALL
    USING (workspace_id = my_workspace_id())
    WITH CHECK (workspace_id = my_workspace_id());

COMMENT ON TABLE workspace_telephony_providers IS 
    'Connected telephony provider accounts (BYO Twilio, etc.).
     Credentials are stored encrypted in provider_secrets table.
     vault_secret_id references provider_secrets.id (not Supabase Vault).';

-- =============================================================================
-- 3. VERIFY PHONE_NUMBERS TABLE HAS TELEPHONY_PROVIDER_ID
-- =============================================================================
-- Add column if it doesn't exist (from V4 schema)

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'phone_numbers' 
        AND column_name = 'telephony_provider_id'
    ) THEN
        ALTER TABLE phone_numbers 
        ADD COLUMN telephony_provider_id UUID 
        REFERENCES workspace_telephony_providers(id) ON DELETE SET NULL;
        
        COMMENT ON COLUMN phone_numbers.telephony_provider_id IS 
            'NULL = platform number (uses env vars). 
             Non-NULL = BYO number (uses connected provider credentials).';
    END IF;
END $$;

-- =============================================================================
-- 4. VERIFICATION QUERIES (Run these to confirm setup)
-- =============================================================================

-- Check provider_secrets table
SELECT 'provider_secrets table' as check_item, 
       COUNT(*) as column_count,
       'Should be 5 columns' as expected
FROM information_schema.columns 
WHERE table_schema = 'public' 
  AND table_name = 'provider_secrets';

-- Check workspace_telephony_providers table  
SELECT 'workspace_telephony_providers table' as check_item,
       COUNT(*) as column_count,
       'Should be 10 columns' as expected
FROM information_schema.columns 
WHERE table_schema = 'public' 
  AND table_name = 'workspace_telephony_providers';

-- Check phone_numbers has the FK column
SELECT 'phone_numbers.telephony_provider_id' as check_item,
       CASE WHEN EXISTS (
           SELECT 1 FROM information_schema.columns 
           WHERE table_name = 'phone_numbers' 
           AND column_name = 'telephony_provider_id'
       ) THEN 'EXISTS' ELSE 'MISSING' END as status;

-- Check RLS policies
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual
FROM pg_policies 
WHERE tablename IN ('provider_secrets', 'workspace_telephony_providers')
ORDER BY tablename, policyname;

-- =============================================================================
-- 5. TEST INSERT (Optional - verify RLS works)
-- =============================================================================
-- Uncomment to test (will only work if running as authenticated user with workspace)

/*
-- This should work for the current user's workspace:
INSERT INTO provider_secrets (id, workspace_id, provider, encrypted_data)
VALUES (
    'test_secret_123', 
    (SELECT id FROM workspaces WHERE owner_id = auth.uid() LIMIT 1),
    'twilio',
    'gAAAAAB...encrypted_data_here...'
)
ON CONFLICT DO NOTHING;

-- Clean up test data
DELETE FROM provider_secrets WHERE id = 'test_secret_123';
*/

-- =============================================================================
-- SUMMARY
-- =============================================================================
-- After running this script, you should have:
--
-- 1. provider_secrets table - Stores encrypted credentials
--    - id: Text primary key
--    - workspace_id: Links to workspace
--    - provider: 'twilio', etc.
--    - encrypted_data: Fernet-encrypted JSON
--    - created_at: Timestamp
--
-- 2. workspace_telephony_providers table - Tracks connected providers
--    - Links to provider_secrets via vault_secret_id
--    - Tracks verification status
--
-- 3. phone_numbers.telephony_provider_id - FK to provider
--    - NULL = platform number
--    - Non-NULL = BYO number
--
-- 4. RLS policies on both tables for security
--
-- NEXT STEPS:
-- 1. Add ENCRYPTION_KEY to backend .env file
-- 2. Restart backend container
-- 3. Test connecting Twilio account in Dashboard
-- =============================================================================
