import os
import psycopg2
import time # Import time module
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Database Configuration ---
# Reads the same .env file as context7_extractor.py
DB_NAME = os.getenv("DB_NAME", "postgres") # Default to 'postgres' database
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
# Read port from .env, clean potential comments, validate, and default
raw_db_port_str = os.getenv("DB_PORT", "54533")
db_port_cleaned = raw_db_port_str.split('#')[0].strip()
DB_PORT = 54533 # Default port
try:
    DB_PORT = int(db_port_cleaned) # Ensure port is an integer
    if not (1024 <= DB_PORT <= 65535): # Basic port range check
         print(f"Warning: DB_PORT '{DB_PORT}' is outside the typical user port range (1024-65535). Using it anyway.")
except ValueError:
    print(f"Error: Invalid value '{db_port_cleaned}' (from '{raw_db_port_str}') found for DB_PORT in .env file. It must be a number.")
    print(f"Using default port {DB_PORT}.")


# --- SQL Script Path ---
SQL_SCRIPT_PATH = "create_context7_docs_table.sql"

# --- Connection Settings ---
INITIAL_CONNECTION_DELAY_SECONDS = 5 # Wait time before first connection attempt

def initialize_database():
    """Connects to the database and executes the initialization SQL script."""
    conn = None
    try:
        # Add a delay to allow the DB container to fully start
        print(f"Waiting {INITIAL_CONNECTION_DELAY_SECONDS} seconds for database service to initialize...")
        time.sleep(INITIAL_CONNECTION_DELAY_SECONDS)

        # Connect to the database
        print(f"Connecting to database '{DB_NAME}' at {DB_HOST}:{DB_PORT}...")
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = False # Ensure commands are run in a transaction
        print("Database connection successful.")

        # Read the SQL script
        print(f"Reading SQL script from: {SQL_SCRIPT_PATH}")
        if not os.path.exists(SQL_SCRIPT_PATH):
            print(f"Error: SQL script not found at {SQL_SCRIPT_PATH}")
            return

        with open(SQL_SCRIPT_PATH, 'r') as f:
            sql_commands = f.read()

        # Execute the SQL script
        print("Executing SQL script to create table and extension...")
        with conn.cursor() as cur:
            cur.execute(sql_commands)

        # Commit the transaction
        conn.commit()
        print("Database initialization successful: 'vector' extension enabled (if not already) and 'context7_docs' table created.")

    except psycopg2.Error as e:
        print(f"Database error during initialization: {e}")
        if conn:
            conn.rollback() # Rollback changes on error
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    if not all([DB_PASSWORD]):
         print("Error: Database password (DB_PASSWORD) not found in .env file.")
         print("Please create or update the .env file with your database credentials.")
    else:
        initialize_database()
