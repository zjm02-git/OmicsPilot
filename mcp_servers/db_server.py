"""数据库 MCP server：把「查用户 / 查分析历史」包装成标准 MCP 工具。

为什么单独一个进程（而不是让 agent 直接 import db.py）：
  MCP 的价值是「标准协议解耦」——Agent 不直接 import 业务库，
  而是通过 stdio + JSON-RPC 连这个 server，server 内部才碰 MySQL。
  以后换成别的数据库、甚至别的语言实现，Agent 侧代码一行都不用改。

用法（作为 MCP server，由 client 用 stdio 启动）：
  python mcp_servers/db_server.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp.server.mcpserver import MCPServer  # noqa: E402

mcp = MCPServer("omics-db")


@mcp.tool()
def get_user_count() -> int:
    """统计系统里注册了多少用户。"""
    # 惰性 import：只有真正调这个工具时才连 MySQL，server 启动本身不碰库
    from db import User, engine
    from sqlalchemy import func
    from sqlalchemy.orm import Session
    with Session(engine) as session:
        return session.query(func.count(User.id)).scalar() or 0


@mcp.tool()
def get_history_by_user(user_id: int) -> str:
    """查某个用户的最近分析历史（最多 10 条，倒序）。"""
    from db import get_history_by_user as _get
    rows = _get(user_id)
    if not rows:
        return f"用户 {user_id} 还没有分析历史。"
    return "\n".join(f"[{r.status}] {r.user_request}（{r.created_at}）" for r in rows[:10])


if __name__ == "__main__":
    # 默认 stdio transport：从 stdin 读 JSON-RPC 请求，把响应写回 stdout
    mcp.run()
