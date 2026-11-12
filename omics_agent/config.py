"""配置：读环境变量 + 造模型 + 造检查点/记忆存储器（工厂模式）。

好处：换模型、换 Key、换存储后端都只改这里，业务代码不用动。
"""

import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# 从项目根目录的 .env 读配置（DEEPSEEK_API_KEY 等）
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")

# 默认本地 PostgreSQL 连接
DEFAULT_DB_URL = "postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable"


def get_model() -> ChatDeepSeek:
    """统一"造模型"的入口。要换模型就改这里。"""
    return ChatDeepSeek(
        model=MODEL_NAME,
        # 关掉思考模式，让回答更快、更省 token
        extra_body={"thinking": {"type": "disabled"}},
    )


def use_postgres() -> bool:
    """是否用 PostgreSQL 持久化。设 USE_POSTGRES=true 才启用；默认内存版开箱即跑。"""
    return os.getenv("USE_POSTGRES", "").strip().lower() in ("1", "true", "yes")


def get_db_url() -> str:
    """数据库连接串，可用 DATABASE_URL 覆盖。"""
    return os.getenv("DATABASE_URL", DEFAULT_DB_URL)


def get_checkpointer():
    """检查点存储器（短期记忆）。默认内存；启用 PG 时用 PostgresSaver，重启不丢。"""
    if use_postgres():
        from psycopg_pool import ConnectionPool
        from psycopg.rows import dict_row
        from langgraph.checkpoint.postgres import PostgresSaver
        # autocommit=True：setup 的迁移里有 CREATE INDEX CONCURRENTLY，不能在事务里跑
        # row_factory=dict_row：查询结果按列名返回字典，PostgresSaver 靠它解析结果
        pool = ConnectionPool(get_db_url(), kwargs={"autocommit": True, "row_factory": dict_row})
        checkpointer = PostgresSaver(pool)    # 传连接池，拿到真正的 PostgresSaver 实例
        checkpointer.setup()  # 幂等：自动建表
        return checkpointer
    return InMemorySaver()


def get_store():
    """长期记忆存储器。默认内存；启用 PG 时用 PostgresStore，重启不丢。"""
    if use_postgres():
        from psycopg_pool import ConnectionPool
        from psycopg.rows import dict_row
        from langgraph.store.postgres import PostgresStore
        # autocommit=True + row_factory=dict_row 的理由同 get_checkpointer
        pool = ConnectionPool(get_db_url(), kwargs={"autocommit": True, "row_factory": dict_row})
        store = PostgresStore(pool)           # 传连接池，拿到真正的 PostgresStore 实例
        store.setup()  # 幂等：自动建表
        return store
    return InMemoryStore()
