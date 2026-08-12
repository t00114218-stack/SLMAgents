# SLM Database Migrator

A lightweight, local CPU-optimized Database Schema Migrator and ORM Generator. It allows developers to diff SQL schemas, generate database migrations, produce SQLAlchemy/SQLModel classes, and verify safety inside an isolated SQLite memory sandbox.

---

## Features

- **Schema Diffing & Migration SQL**: Generates exact DDL migration queries (e.g. `ALTER TABLE`, `CREATE INDEX`) to reconcile schemas.
- **ORM Declarative Mapping**: Auto-generates SQLAlchemy/SQLModel classes representing the target database schema.
- **SQLite Sandbox Verification**: Runs generated DDL statements inside an in-memory sandboxed database connection to catch syntax errors or constraint violations before applying.
- **100% Offline & Local**: Relies on local structural matching and offline verification loops.

---

## Installation

```bash
pip install -e ./slm_db_migration
```

---

## API Reference

### `SLMDBMigrator`

```python
from slm_db_migration import SLMDBMigrator

migrator = SLMDBMigrator()
```

#### `generate_migration(from_schema_sql: str, to_schema_sql: str) -> dict`
Generates migration SQL, SQLAlchemy classes, and validates execution inside a sandbox database.
- **Arguments**:
  - `from_schema_sql` (str): DDL for the source schema.
  - `to_schema_sql` (str): DDL for the target schema.
- **Returns**:
  - `dict`:
    ```python
    {
        "success": True/False,
        "migration_sql": str,      # Generated DDL Migration statements
        "sqlalchemy_code": str,     # SQLAlchemy model classes
        "sandbox_result": str       # Status message from SQLite sandbox verification
    }
    ```

---

## Usage Example

```python
from slm_db_migration import SLMDBMigrator

migrator = SLMDBMigrator()

source_ddl = "CREATE TABLE users (id INT PRIMARY KEY, name TEXT);"
target_ddl = "CREATE TABLE users (id INT PRIMARY KEY, name TEXT, email TEXT);"

result = migrator.generate_migration(source_ddl, target_ddl)

print(f"Success: {result['success']}")
print(f"Migration DDL:\n{result['migration_sql']}")
print(f"SQLAlchemy Models:\n{result['sqlalchemy_code']}")
print(f"Sandbox Verification: {result['sandbox_result']}")
```

### Input & Output Example

#### Input (Source Schema):
```sql
CREATE TABLE users (id INT PRIMARY KEY, name TEXT);
```

#### Input (Target Schema):
```sql
CREATE TABLE users (id INT PRIMARY KEY, name TEXT, email TEXT);
```

#### Output:
```json
{
  "success": true,
  "migration_sql": "ALTER TABLE users ADD COLUMN email TEXT;",
  "sqlalchemy_code": "from sqlalchemy import Column, Integer, String\nfrom sqlalchemy.orm import declarative_base\n...\nclass User(Base):\n    __tablename__ = 'users'\n    id = Column(Integer, primary_key=True)\n    name = Column(String)\n    email = Column(String)",
  "sandbox_result": "Migration verified successfully in SQLite sandbox."
}
```
