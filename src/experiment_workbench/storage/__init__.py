"""永続化層: プロジェクトフォルダ・SQLite メタデータ（Phase 1 で Parquet を追加予定）。

利用側は ``from experiment_workbench.storage import ProjectStore`` のように、
このパッケージから公開 API を import する。このパッケージは Qt に依存しない。
"""

from experiment_workbench.storage.database import (
    ENTITY_TABLES,
    EntityNotFoundError,
    IntegrityViolationError,
    MetadataDatabase,
    MigrationError,
    SchemaVersionError,
    StorageError,
    latest_schema_version,
)
from experiment_workbench.storage.project_store import (
    METADATA_DB_NAME,
    PROJECT_FILE_NAME,
    ProjectDirectory,
    ProjectStore,
    ProjectStoreError,
)

__all__ = [
    "ENTITY_TABLES",
    "METADATA_DB_NAME",
    "PROJECT_FILE_NAME",
    "EntityNotFoundError",
    "IntegrityViolationError",
    "MetadataDatabase",
    "MigrationError",
    "ProjectDirectory",
    "ProjectStore",
    "ProjectStoreError",
    "SchemaVersionError",
    "StorageError",
    "latest_schema_version",
]
