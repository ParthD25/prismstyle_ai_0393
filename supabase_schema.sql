-- PrismStyle AI - Supabase Database Schema
-- Run this SQL in your Supabase SQL Editor to set up the database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================
-- USERS TABLE
-- Stores user profiles with style preferences
-- =============================================
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email TEXT UNIQUE NOT NULL,
  full_name TEXT,
  avatar_url TEXT,
  gender TEXT CHECK (gender IN ('male', 'female', 'non-binary', 'prefer_not_to_say')),
  style_preferences JSONB DEFAULT '{}',
  favorite_colors JSONB DEFAULT '[]',
  body_type TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster email lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- =============================================
-- CLOTHING ITEMS TABLE
-- Stores wardrobe items with AI-detected attributes
-- =============================================
CREATE TABLE IF NOT EXISTS clothing_items (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  name TEXT,
  photo_url TEXT,
  category TEXT NOT NULL CHECK (category IN ('Tops', 'Bottoms', 'Dresses', 'Shoes', 'Accessories', 'Outerwear', 'Activewear', 'Formal')),
  subcategory TEXT,
  color TEXT,
  secondary_color TEXT,
  pattern TEXT DEFAULT 'Solid',
  material TEXT,
  style TEXT,
  season TEXT[] DEFAULT '{}',
  occasion TEXT[] DEFAULT '{}',
  brand TEXT,
  size TEXT,
  ai_confidence FLOAT CHECK (ai_confidence >= 0 AND ai_confidence <= 1),
  ai_predictions JSONB DEFAULT '{}',
  color_analysis JSONB DEFAULT '{}',
  tags TEXT[] DEFAULT '{}',
  is_favorite BOOLEAN DEFAULT FALSE,
  wear_count INTEGER DEFAULT 0,
  last_worn_at TIMESTAMP WITH TIME ZONE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_clothing_items_user_id ON clothing_items(user_id);
CREATE INDEX IF NOT EXISTS idx_clothing_items_category ON clothing_items(category);
CREATE INDEX IF NOT EXISTS idx_clothing_items_color ON clothing_items(color);
CREATE INDEX IF NOT EXISTS idx_clothing_items_is_favorite ON clothing_items(is_favorite);

-- =============================================
-- OUTFITS TABLE
-- Stores generated and saved outfit combinations
-- =============================================
CREATE TABLE IF NOT EXISTS outfits (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  name TEXT,
  occasion TEXT,
  weather_condition TEXT,
  temperature_range TEXT,
  outfit_data JSONB NOT NULL DEFAULT '{}',
  item_ids UUID[] DEFAULT '{}',
  compatibility_score FLOAT CHECK (compatibility_score >= 0 AND compatibility_score <= 100),
  style_notes TEXT,
  is_saved BOOLEAN DEFAULT FALSE,
  is_shared BOOLEAN DEFAULT FALSE,
  times_worn INTEGER DEFAULT 0,
  last_worn_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_outfits_user_id ON outfits(user_id);
CREATE INDEX IF NOT EXISTS idx_outfits_is_saved ON outfits(is_saved);
CREATE INDEX IF NOT EXISTS idx_outfits_occasion ON outfits(occasion);

-- =============================================
-- FRIEND RELATIONSHIPS TABLE
-- Manages social connections between users
-- =============================================
CREATE TABLE IF NOT EXISTS friend_relationships (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  friend_id UUID REFERENCES users(id) ON DELETE CASCADE,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'rejected', 'blocked')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, friend_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_friend_relationships_user_id ON friend_relationships(user_id);
CREATE INDEX IF NOT EXISTS idx_friend_relationships_friend_id ON friend_relationships(friend_id);
CREATE INDEX IF NOT EXISTS idx_friend_relationships_status ON friend_relationships(status);

-- =============================================
-- OUTFIT FEEDBACK TABLE
-- Community feedback and ratings on shared outfits
-- =============================================
CREATE TABLE IF NOT EXISTS outfit_feedback (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  outfit_id UUID REFERENCES outfits(id) ON DELETE CASCADE,
  from_user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  rating INTEGER CHECK (rating >= 1 AND rating <= 5),
  comment TEXT,
  reaction TEXT CHECK (reaction IN ('love', 'like', 'fire', 'cool', 'meh')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(outfit_id, from_user_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_outfit_feedback_outfit_id ON outfit_feedback(outfit_id);
CREATE INDEX IF NOT EXISTS idx_outfit_feedback_from_user_id ON outfit_feedback(from_user_id);

-- =============================================
-- STYLE HISTORY TABLE
-- Tracks outfit wearing history for recommendations
-- =============================================
CREATE TABLE IF NOT EXISTS style_history (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  outfit_id UUID REFERENCES outfits(id) ON DELETE SET NULL,
  item_ids UUID[] DEFAULT '{}',
  worn_date DATE NOT NULL,
  weather_conditions JSONB DEFAULT '{}',
  occasion TEXT,
  user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_style_history_user_id ON style_history(user_id);
CREATE INDEX IF NOT EXISTS idx_style_history_worn_date ON style_history(worn_date);

-- =============================================
-- NOTIFICATIONS TABLE
-- Stores user notifications
-- =============================================
CREATE TABLE IF NOT EXISTS notifications (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  type TEXT NOT NULL CHECK (type IN ('outfit', 'weather', 'social', 'tips', 'system')),
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  data JSONB DEFAULT '{}',
  is_read BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON notifications(is_read);

-- =============================================
-- FCM TOKENS TABLE
-- Stores Firebase Cloud Messaging tokens for push notifications
-- =============================================
CREATE TABLE IF NOT EXISTS fcm_tokens (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  token TEXT NOT NULL,
  device_info JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, token)
);

-- Index
CREATE INDEX IF NOT EXISTS idx_fcm_tokens_user_id ON fcm_tokens(user_id);

-- =============================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =============================================

-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE clothing_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE outfits ENABLE ROW LEVEL SECURITY;
ALTER TABLE friend_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE outfit_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE style_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE fcm_tokens ENABLE ROW LEVEL SECURITY;

-- Users policies
CREATE POLICY "Users can view their own profile" ON users
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON users
  FOR UPDATE USING (auth.uid() = id);

-- Clothing items policies
CREATE POLICY "Users can view their own clothing items" ON clothing_items
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own clothing items" ON clothing_items
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own clothing items" ON clothing_items
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own clothing items" ON clothing_items
  FOR DELETE USING (auth.uid() = user_id);

-- Outfits policies
CREATE POLICY "Users can view their own outfits" ON outfits
  FOR SELECT USING (auth.uid() = user_id OR is_shared = TRUE);

CREATE POLICY "Users can insert their own outfits" ON outfits
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own outfits" ON outfits
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own outfits" ON outfits
  FOR DELETE USING (auth.uid() = user_id);

-- Friend relationships policies
CREATE POLICY "Users can view their friend relationships" ON friend_relationships
  FOR SELECT USING (auth.uid() = user_id OR auth.uid() = friend_id);

CREATE POLICY "Users can create friend requests" ON friend_relationships
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their friend relationships" ON friend_relationships
  FOR UPDATE USING (auth.uid() = user_id OR auth.uid() = friend_id);

-- Outfit feedback policies
CREATE POLICY "Users can view feedback on accessible outfits" ON outfit_feedback
  FOR SELECT USING (TRUE);

CREATE POLICY "Users can insert feedback" ON outfit_feedback
  FOR INSERT WITH CHECK (auth.uid() = from_user_id);

-- Style history policies
CREATE POLICY "Users can view their own style history" ON style_history
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own style history" ON style_history
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Notifications policies
CREATE POLICY "Users can view their own notifications" ON notifications
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own notifications" ON notifications
  FOR UPDATE USING (auth.uid() = user_id);

-- FCM tokens policies
CREATE POLICY "Users can manage their own FCM tokens" ON fcm_tokens
  FOR ALL USING (auth.uid() = user_id);

-- =============================================
-- STORAGE BUCKETS
-- Run these in the Supabase Storage section
-- =============================================
-- Create buckets:
-- 1. clothing-images (public)
-- 2. outfit-images (public)
-- 3. profile-avatars (public)

-- =============================================
-- FUNCTIONS & TRIGGERS
-- =============================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_users_updated_at
  BEFORE UPDATE ON users
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_clothing_items_updated_at
  BEFORE UPDATE ON clothing_items
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_outfits_updated_at
  BEFORE UPDATE ON outfits
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_friend_relationships_updated_at
  BEFORE UPDATE ON friend_relationships
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fcm_tokens_updated_at
  BEFORE UPDATE ON fcm_tokens
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================
-- REALTIME SUBSCRIPTIONS
-- Enable realtime for specific tables
-- =============================================
-- Run in Supabase Dashboard > Database > Replication:
-- Enable realtime for: clothing_items, outfits, outfit_feedback, notifications
