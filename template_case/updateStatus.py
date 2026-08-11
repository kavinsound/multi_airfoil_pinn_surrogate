import sqlite3
from pathlib import Path
import os
import sys

id = Path(__file__).resolve().parent.stem

def update_db(option):
    db = Path("../../job_status.db") #hard coding path and name but ok

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        """
            UPDATE job_stages 
            SET stage = ? 
            WHERE job_id = ?
        """,
        (option, id),
    )

    conn.commit()
    conn.close()



if __name__ == "__main__":
    option = int(sys.argv[1])

    update_db(option)