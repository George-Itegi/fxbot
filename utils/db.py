"""
MySQL Database Connection Pool
================================
Shared MySQL connection for all modules using XAMPP MySQL.
Database: apex_trader
"""

import mysql.connector
from mysql.connector import Error as MySQLError
from mysql.connector import pooling

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
from utils.logger import setup_logger

logger = setup_logger("utils.db")


_connection_pool = None


def get_pool():
    """Get or create the MySQL connection pool (singleton)."""
    global _connection_pool
    if _connection_pool is None:
        try:
            _connection_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="deriv_bot_pool",
                pool_size=5,
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                autocommit=True,
            )
            logger.info(
                f"MySQL pool created: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            )
        except MySQLError as e:
            logger.error(f"MySQL connection failed: {e}")
            raise
    return _connection_pool


def get_connection():
    """Get a connection from the pool."""
    pool = get_pool()
    try:
        return pool.get_connection()
    except MySQLError as e:
        logger.error(f"Failed to get connection from pool: {e}")
        raise


def execute_query(query: str, params: tuple = None, fetch: str = None):
    """
    Execute a query with auto connection management.
    
    Args:
        query: SQL query string
        params: Tuple of parameters
        fetch: "one", "all", or None (for INSERT/UPDATE/DELETE)
    
    Returns:
        Fetched rows or None
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        
        if fetch == "one":
            return cursor.fetchone()
        elif fetch == "all":
            return cursor.fetchall()
        else:
            return None
    except MySQLError as e:
        logger.error(f"Query failed: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def test_connection():
    """Test the MySQL connection."""
    try:
        result = execute_query("SELECT 1 AS ok", fetch="one")
        if result and result.get("ok") == 1:
            logger.info("MySQL connection test: OK")
            return True
    except Exception as e:
        logger.error(f"MySQL connection test FAILED: {e}")
        return False
