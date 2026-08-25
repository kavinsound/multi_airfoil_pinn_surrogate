import sqlite3
from pathlib import Path
STAGE_MAP = {0: "Initialized", 1: "Meshing", 2: "SimpleFoam Solve", 3: "Reading data", 4: "Completed"}

def getCounts(db_path="job_status.db"):

    conn = sqlite3.connect(Path(db_path))
    cursor = conn.cursor()

    query = """
        SELECT stage, COUNT(*) as count
        FROM job_stages
        GROUP BY stage
    """

    try:
        cursor.execute(query)
        results = cursor.fetchall()

        print("\n" + "=" * 36)
        print(f"{'Case Job Status':^36}")
        print("=" * 36)
        print(f"{'Stage':<22} | {'Count':<5}")
        print("-" * 36)

        total_jobs = 0

        for stage_num, count in results:
            stage_name = STAGE_MAP.get(stage_num, f"Unknown ({stage_num})")

            print(f"{stage_name:<22} | {count:<5}")
            total_jobs += count

        print("-" * 36)
        print(f"{'Total Jobs':<22} | {total_jobs:<5}")
        print("=" * 36)

    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")

    finally:
        conn.close()



if __name__ == "__main__":
    getCounts()