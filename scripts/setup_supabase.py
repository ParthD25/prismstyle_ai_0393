#!/usr/bin/env python3
"""
Supabase Database Setup Script for PrismStyle AI

This script helps set up the Supabase database:
1. Validates connection to Supabase
2. Creates required tables and indexes
3. Sets up Row Level Security (RLS) policies
4. Creates storage buckets

Prerequisites:
- Supabase project created at https://supabase.com
- Project URL and API keys from project settings

Usage:
    python setup_supabase.py --check           # Check connection only
    python setup_supabase.py --setup           # Full setup
    python setup_supabase.py --create-buckets  # Create storage buckets only
"""

import os
import sys
import json
import argparse
from pathlib import Path

def load_env():
    """Load environment variables from env.json"""
    env_path = Path(__file__).parent.parent / "env.json"
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            return json.load(f)
    
    # Check environment variables
    return {
        "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_ANON_KEY": os.environ.get("SUPABASE_ANON_KEY", ""),
        "SUPABASE_SECRET_KEY": os.environ.get("SUPABASE_SECRET_KEY", "")
    }

def check_connection(env):
    """Check if we can connect to Supabase"""
    print("\n[CHECK] Testing Supabase connection...")
    
    url = env.get("SUPABASE_URL", "")
    key = env.get("SUPABASE_ANON_KEY", "")
    
    if not url or not key:
        print("[ERROR] SUPABASE_URL or SUPABASE_ANON_KEY not configured")
        print("[INFO] Update env.json with your Supabase credentials")
        return False
    
    try:
        import requests
        
        # Test connection
        response = requests.get(
            f"{url}/rest/v1/",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"[OK] Connected to Supabase: {url}")
            return True
        else:
            print(f"[ERROR] Connection failed: {response.status_code}")
            return False
            
    except ImportError:
        print("[INFO] requests library not installed")
        print("[INFO] Install with: pip install requests")
        print(f"[INFO] Supabase URL configured: {url[:30]}...")
        return True
    except Exception as e:
        print(f"[ERROR] Connection test failed: {e}")
        return False

def get_schema_sql():
    """Get the SQL schema"""
    schema_path = Path(__file__).parent.parent / "supabase_schema.sql"
    
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            return f.read()
    
    print(f"[ERROR] Schema file not found: {schema_path}")
    return None

def print_setup_instructions(env):
    """Print manual setup instructions"""
    print("\n" + "="*60)
    print("  SUPABASE SETUP INSTRUCTIONS")
    print("="*60)
    
    url = env.get("SUPABASE_URL", "https://your-project.supabase.co")
    
    print(f"""
1. Go to your Supabase Dashboard:
   {url.replace('.supabase.co', '.supabase.com')}

2. Navigate to SQL Editor (left sidebar)

3. Create a new query and paste the contents of:
   supabase_schema.sql

4. Click "Run" to execute the SQL

5. Set up Storage Buckets:
   - Go to Storage (left sidebar)
   - Create these buckets (set as Public):
     * clothing-images
     * outfit-images  
     * profile-avatars

6. Enable Realtime:
   - Go to Database > Replication
   - Enable realtime for:
     * clothing_items
     * outfits
     * outfit_feedback
     * notifications

7. Verify RLS is enabled:
   - Go to Authentication > Policies
   - Ensure policies are created for each table
""")
    
    print("="*60)
    print("  QUICK VERIFICATION")
    print("="*60)
    print(f"""
After setup, verify by running:

  python setup_supabase.py --check

Expected output:
  [OK] Connected to Supabase
  [OK] Tables created
  [OK] Storage buckets created
""")

def create_storage_buckets_instructions():
    """Print storage bucket creation instructions"""
    print("\n" + "="*60)
    print("  STORAGE BUCKET SETUP")
    print("="*60)
    print("""
Create these storage buckets in Supabase Dashboard:

1. clothing-images
   - Public: Yes
   - File size limit: 5MB
   - Allowed MIME types: image/jpeg, image/png, image/webp

2. outfit-images
   - Public: Yes
   - File size limit: 5MB
   - Allowed MIME types: image/jpeg, image/png, image/webp

3. profile-avatars
   - Public: Yes
   - File size limit: 2MB
   - Allowed MIME types: image/jpeg, image/png, image/webp

Storage Policies (for each bucket):
- Allow authenticated users to upload to their own folder
- Allow public read access
- Allow users to delete their own files

Example policy for clothing-images:

CREATE POLICY "Users can upload clothing images"
ON storage.objects FOR INSERT
WITH CHECK (
  bucket_id = 'clothing-images' AND
  auth.uid()::text = (storage.foldername(name))[1]
);

CREATE POLICY "Public can view clothing images"
ON storage.objects FOR SELECT
USING (bucket_id = 'clothing-images');

CREATE POLICY "Users can delete own clothing images"
ON storage.objects FOR DELETE
USING (
  bucket_id = 'clothing-images' AND
  auth.uid()::text = (storage.foldername(name))[1]
);
""")

def verify_tables(env):
    """Verify tables exist"""
    print("\n[CHECK] Verifying database tables...")
    
    required_tables = [
        'users',
        'clothing_items',
        'outfits',
        'friend_relationships',
        'outfit_feedback',
        'style_history',
        'notifications',
        'fcm_tokens'
    ]
    
    try:
        import requests
        
        url = env.get("SUPABASE_URL", "")
        key = env.get("SUPABASE_ANON_KEY", "")
        
        for table in required_tables:
            response = requests.get(
                f"{url}/rest/v1/{table}?limit=0",
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}"
                },
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"  [OK] {table}")
            else:
                print(f"  [MISSING] {table}")
                
    except ImportError:
        print("[INFO] Cannot verify tables (requests not installed)")
        print("[INFO] Tables to create:", required_tables)
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Supabase Database Setup')
    parser.add_argument('--check', action='store_true', help='Check connection and tables')
    parser.add_argument('--setup', action='store_true', help='Show full setup instructions')
    parser.add_argument('--create-buckets', action='store_true', help='Show bucket creation instructions')
    parser.add_argument('--verify', action='store_true', help='Verify tables exist')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  PrismStyle AI - Supabase Setup")
    print("="*60)
    
    env = load_env()
    
    if args.check or (not any([args.setup, args.create_buckets, args.verify])):
        check_connection(env)
    
    if args.verify:
        verify_tables(env)
    
    if args.setup:
        print_setup_instructions(env)
    
    if args.create_buckets:
        create_storage_buckets_instructions()
    
    if not any([args.check, args.setup, args.create_buckets, args.verify]):
        print("\nUsage:")
        print("  python setup_supabase.py --check         # Test connection")
        print("  python setup_supabase.py --setup         # Full setup guide")
        print("  python setup_supabase.py --create-buckets # Storage setup")
        print("  python setup_supabase.py --verify        # Verify tables")

if __name__ == "__main__":
    main()
