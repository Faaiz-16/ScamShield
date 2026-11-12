from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

import app.models  # noqa: F401
from app.core.config import BASE_DIR
from app.core.database import Base


@pytest.fixture
def alembic_config(tmp_path: Path) -> tuple[Config, str]:
    url = f"sqlite:///{tmp_path / 'migrations.db'}"
    config = Config(str(BASE_DIR / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    return config, url


def test_migrations_match_models(alembic_config: tuple[Config, str]) -> None:
    config, url = alembic_config
    command.upgrade(config, "head")
    engine = create_engine(url)
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []
    engine.dispose()


def test_downgrade_removes_all_tables(alembic_config: tuple[Config, str]) -> None:
    config, url = alembic_config
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    engine = create_engine(url)
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    engine.dispose()


def test_mysql_sql_uses_utf8mb4(alembic_config: tuple[Config, str], capsys) -> None:
    config, _ = alembic_config
    config.set_main_option("sqlalchemy.url", "mysql+pymysql://user:pass@localhost/scamshield")
    command.upgrade(config, "head", sql=True)
    sql = capsys.readouterr().out
    assert sql.count("CHARSET=utf8mb4") == 2
    assert "ON DELETE CASCADE" in sql
