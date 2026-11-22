"""数据库 MCP server：把「查用户 / 查分析历史」包装成标准 MCP 工具。

为什么单独一个进程（而不是让 agent 直接 import db.py）：
  MCP 的价值是「标准协议解耦」——Agent 不直接 import 业务库，
  而是通过 stdio + JSON-RPC 连这个 server，server 内部才碰 MySQL。
  以后换成别的数据库、甚至别的语言实现，Agent 侧代码一行都不用改。

用法（作为 MCP server，由 client 用 stdio 启动）：
  python mcp_servers/db_server.py

关于入口 API（这里踩过一次坑，值得记下来）：
  这个类在 MCP 官方 Python SDK 里**改过名字**，所以不能写死一个：
    mcp < 2   ：叫 FastMCP，位置 mcp.server.fastmcp
    mcp >= 2  ：又改回 MCPServer，位置 mcp.server.mcpserver
               （官方迁移指南：fastmcp-renamed-to-mcpserver）
  所以只写其中任何一个，换个环境就会 ModuleNotFoundError，
  表现为「仓库里有这个 server、线上却起不来」。下面做版本自适应导入。
  两个类的样板活一样（装饰器登记工具、JSON-RPC 收发、stdio 读写循环），
  接口也一致（.tool() / .run()），所以只需要换类名。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _MCPServer  # noqa: E402
except ImportError:  # mcp < 2
    from mcp.server.fastmcp import FastMCP as _MCPServer  # noqa: E402

# 名字会出现在 initialize 响应的 serverInfo 里，client 侧排障时一眼能认出来
mcp = _MCPServer("omics-db")


@mcp.tool()
def get_user_count() -> int:
    """统计系统里注册了多少用户。"""
    # 惰性 import：只有真正调这个工具时才连 MySQL，server 启动本身不碰库。
    # 这一点很关键——server 起来时不依赖数据库可用，MCP 握手不会被 DB 拖挂。
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
    # 不带参数默认就是 stdio transport：从 stdin 读 JSON-RPC 请求、把响应写回 stdout。
    # 注意：业务日志一定不要 print 到 stdout——那会污染协议流，client 解析会直接报错；
    # 要打日志就写 stderr（logging 的 StreamHandler 默认就是 stderr）。
    mcp.run()
