"""Database layer using Neon PostgreSQL via psycopg2.

Set the DATABASE_URL environment variable to your Neon connection string:
    DATABASE_URL=postgresql://user:password@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
"""
import os
import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_kY8nABf3TXaJ@ep-tiny-bread-ank4n1m9-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)


class _Row(dict):
    """Dict wrapper that supports both key access (row['key']) and attribute access (row.key)."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)


class NeonDB:
    """
    Wraps a psycopg2 connection + cursor and mimics the sqlite3 interface
    used throughout the codebase:
        db.execute(sql, params)   -> returns self; sets .lastrowid on INSERT … RETURNING id
        db.executescript(sql)     -> splits on ';' and runs each DDL statement
        db.fetchone()             -> _Row or None
        db.fetchall()             -> list[_Row]
        db.commit()
        db.rollback()
        db.close()

    Key behaviours vs plain sqlite3:
      - ? placeholders are automatically converted to %s for psycopg2.
      - Every INSERT must end with RETURNING id so that .lastrowid is populated.
      - Any exception inside execute() triggers an automatic rollback so the
        connection is never left in an aborted-transaction state.
    """

    def __init__(self):
        self._conn = psycopg2.connect(DATABASE_URL)
        self._conn.autocommit = False
        self._cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        self.lastrowid = None
        self._returning_consumed = False  # True after we've read the RETURNING id row

    # ── Core execute ──────────────────────────────────────────────────────────

    def execute(self, sql, params=None):
        """Execute one SQL statement.

        - Converts ? → %s automatically.
        - If the statement contains RETURNING id, captures the returned id into
          self.lastrowid.
        - On any psycopg2 exception, rolls back the transaction before re-raising
          so the connection is left in a clean state.
        """
        pg_sql = sql.replace('?', '%s')
        self._returning_consumed = False
        self.lastrowid = None

        try:
            self._cur.execute(pg_sql, params)
        except Exception:
            # Roll back aborted transaction so subsequent queries can proceed.
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise

        # If this was an INSERT … RETURNING id, fetch and store the id now.
        if self._cur.description:
            col_names = [d[0] for d in self._cur.description]
            if col_names == ['id']:
                row = self._cur.fetchone()
                if row:
                    self.lastrowid = row['id']
                self._returning_consumed = True  # cursor result set is now empty

        return self

    # ── Bulk DDL ──────────────────────────────────────────────────────────────

    def executescript(self, sql):
        """Run multiple semicolon-separated DDL statements (used by init_db).

        Uses a plain (non-RealDict) cursor to avoid overhead, commits at the end.
        """
        statements = [s.strip() for s in sql.split(';') if s.strip()]
        try:
            cur = self._conn.cursor()
            for stmt in statements:
                cur.execute(stmt)
            cur.close()
            self._conn.commit()
        except Exception:
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise

    # ── Fetch helpers ─────────────────────────────────────────────────────────

    def fetchone(self):
        """Return the next row as a _Row dict, or None."""
        if self._returning_consumed:
            return None
        row = self._cur.fetchone()
        return _Row(row) if row else None

    def fetchall(self):
        """Return all remaining rows as a list of _Row dicts."""
        if self._returning_consumed:
            return []
        rows = self._cur.fetchall()
        return [_Row(r) for r in rows]

    # ── Transaction / lifecycle ───────────────────────────────────────────────

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        try:
            self._cur.close()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_db():
    """Return a fresh NeonDB instance (new connection + cursor)."""
    return NeonDB()


def init_db():
    """Create all application tables if they do not already exist."""
    db = get_db()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id              SERIAL PRIMARY KEY,
        username        TEXT UNIQUE NOT NULL,
        password_hash   TEXT NOT NULL,
        role            TEXT NOT NULL,
        full_name       TEXT,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        last_login      TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS uploads (
        id              SERIAL PRIMARY KEY,
        filename        TEXT,
        upload_type     TEXT,
        uploaded_by     INTEGER,
        uploaded_at     TIMESTAMPTZ DEFAULT NOW(),
        row_count       INTEGER,
        status          TEXT DEFAULT 'pending',
        notes           TEXT
    );

    CREATE TABLE IF NOT EXISTS ml_runs (
        id              SERIAL PRIMARY KEY,
        run_type        TEXT,
        triggered_by    INTEGER,
        started_at      TIMESTAMPTZ DEFAULT NOW(),
        finished_at     TIMESTAMPTZ,
        status          TEXT DEFAULT 'running',
        wmape_7d        REAL,
        wmape_30d       REAL,
        log_text        TEXT
    );

    CREATE TABLE IF NOT EXISTS forecasts (
        id              SERIAL PRIMARY KEY,
        ml_run_id       INTEGER,
        product_id      INTEGER,
        warehouse_id    INTEGER,
        category        TEXT,
        forecast_7d     REAL,
        forecast_30d    REAL,
        daily_rate      REAL,
        confidence      REAL,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        month_label     TEXT
    );

    CREATE TABLE IF NOT EXISTS recommendations (
        id              SERIAL PRIMARY KEY,
        role            TEXT,
        rec_type        TEXT,
        title           TEXT,
        detail_json     TEXT,
        confidence      REAL,
        priority        TEXT DEFAULT 'P2',
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        status          TEXT DEFAULT 'pending',
        decided_by      INTEGER,
        decided_at      TIMESTAMPTZ,
        decision_note   TEXT,
        modified_json   TEXT
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id              SERIAL PRIMARY KEY,
        target_role     TEXT,
        source_role     TEXT,
        alert_type      TEXT,
        severity        TEXT,
        title           TEXT,
        body            TEXT,
        detail_json     TEXT,
        is_read         INTEGER DEFAULT 0,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        expires_at      TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS route_events (
        id              SERIAL PRIMARY KEY,
        origin          TEXT,
        destination     TEXT,
        mode            TEXT,
        event_type      TEXT,
        new_cost        REAL,
        reason          TEXT,
        created_by      INTEGER,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        resolved_at     TIMESTAMPTZ,
        is_active       INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS month_close (
        id              SERIAL PRIMARY KEY,
        month_label     TEXT,
        closed_by       INTEGER,
        closed_at       TIMESTAMPTZ DEFAULT NOW(),
        total_forecast  REAL,
        total_actual    REAL,
        wmape           REAL,
        notes           TEXT
    )
    """)
    db.close()