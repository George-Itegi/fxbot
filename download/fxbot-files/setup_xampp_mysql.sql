-- =============================================================
-- APEX TRADER v4.1 — Database Setup for XAMPP MySQL
-- Run this in XAMPP's phpMyAdmin or MySQL shell
-- =============================================================

-- Step 1: Create the database
CREATE DATABASE IF NOT EXISTS apex_trader
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- Step 2: Use the database
USE apex_trader;

-- Step 3: Create tables (bot creates them automatically on startup,
-- but you can create them manually here if preferred)
-- Tables: trades, signals, market_snapshots, strategy_performance
-- All created automatically by db_manager.init_db()
-- Just creating the database is enough.

-- Verify:
SELECT 'Database apex_trader ready!' AS status;
