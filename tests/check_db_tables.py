# tests/check_db_tables.py
import sqlite3

def main():
    con = sqlite3.connect("database/alphainsights.db")
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    con.close()
    print(rows)

if __name__ == "__main__":
    main()
