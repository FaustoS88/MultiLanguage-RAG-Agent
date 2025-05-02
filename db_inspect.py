#!/usr/bin/env python3
"""
Database Inspector for Context7 Documentation Extractor

Provides utilities to inspect and manage the data in the context7_docs table.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# --- Database Configuration ---
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "54533")

def get_db_connection():
    """Establishes and returns a database connection."""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1) # Exit if connection fails
    except Exception as e:
        print(f"An unexpected error occurred during connection: {e}")
        sys.exit(1)

def count_entries(conn):
    """Counts the total number of snippets in the database."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM context7_docs")
            count = cur.fetchone()[0]
            print(f"Database contains {count} documentation snippets.")
    except psycopg2.Error as e:
        print(f"Error counting entries: {e}")

def list_snippets(conn, limit=20):
    """Lists snippets with their library_id, title, and source_url."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, library_id, title, source_url FROM context7_docs ORDER BY library_id, title LIMIT %s",
                (limit,)
            )
            rows = cur.fetchall()
            print(f"\nShowing first {len(rows)} snippets:")
            print("-" * 40)
            for row in rows:
                print(f"ID: {row[0]}")
                print(f"Library: {row[1]}")
                print(f"Title: {row[2]}")
                print(f"Source: {row[3]}")
                print("-" * 40)
    except psycopg2.Error as e:
        print(f"Error listing snippets: {e}")

def view_snippet(conn, snippet_id):
    """Displays the full details of a specific snippet by its UUID."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, library_id, title, description, source_url, language, code FROM context7_docs WHERE id = %s",
                (snippet_id,)
            )
            row = cur.fetchone()
            if not row:
                print(f"No snippet found with ID: {snippet_id}")
                return

            print("\n--- Snippet Details ---")
            print(f"ID: {row[0]}")
            print(f"Library ID: {row[1]}")
            print(f"Title: {row[2]}")
            print(f"Description: {row[3]}")
            print(f"Source URL: {row[4]}")
            print(f"Language: {row[5]}")
            print("\nCode/Content:")
            print("-" * 60)
            print(row[6])
            print("-" * 60)

    except psycopg2.Error as e:
        print(f"Error viewing snippet {snippet_id}: {e}")
    except Exception as e:
         print(f"An unexpected error occurred viewing snippet {snippet_id}: {e}")


def delete_all_snippets(conn):
    """Deletes all snippets from the table after confirmation."""
    confirm = input("Are you sure you want to delete ALL snippets from the 'context7_docs' table? (yes/no): ")
    if confirm.lower() == 'yes':
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM context7_docs")
                count = cur.rowcount
                conn.commit()
                print(f"Successfully deleted {count} snippets.")
        except psycopg2.Error as e:
            print(f"Error deleting all snippets: {e}")
            conn.rollback()
    else:
        print("Deletion cancelled.")

def delete_by_library(conn, library_id):
    """Deletes all snippets for a specific library_id after confirmation."""
    confirm = input(f"Are you sure you want to delete ALL snippets for library_id '{library_id}'? (yes/no): ")
    if confirm.lower() == 'yes':
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM context7_docs WHERE library_id = %s", (library_id,))
                count = cur.rowcount
                conn.commit()
                print(f"Successfully deleted {count} snippets for library_id '{library_id}'.")
        except psycopg2.Error as e:
            print(f"Error deleting snippets for library {library_id}: {e}")
            conn.rollback()
    else:
        print("Deletion cancelled.")


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="Inspect and manage the Context7 documentation database.")
    parser.add_argument('--count', action='store_true', help='Count total snippets in the database.')
    parser.add_argument('--list', type=int, nargs='?', const=20, metavar='LIMIT', help='List snippets (default limit 20).')
    parser.add_argument('--view', type=str, metavar='SNIPPET_ID', help='View a specific snippet by its UUID.')
    parser.add_argument('--delete-all', action='store_true', help='Delete ALL snippets from the database (requires confirmation).')
    parser.add_argument('--delete-library', type=str, metavar='LIBRARY_ID', help='Delete all snippets for a specific library_id (requires confirmation).')

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    if not DB_PASSWORD:
        print("Error: Database password (DB_PASSWORD) not found in .env file.")
        sys.exit(1)

    conn = get_db_connection()
    if not conn:
        sys.exit(1) # Exit if connection failed in get_db_connection

    try:
        if args.count:
            count_entries(conn)
        if args.list is not None: # Handles both --list and --list N
            list_snippets(conn, args.list)
        if args.view:
            view_snippet(conn, args.view)
        if args.delete_all:
            delete_all_snippets(conn)
        if args.delete_library:
            delete_by_library(conn, args.delete_library)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
