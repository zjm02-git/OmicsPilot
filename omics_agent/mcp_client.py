"""MCP 客户端桥接：把 MCP server 的工具「转成」LangChain 同步工具。

为什么要有这个文件：
  agent.py 的 ReAct 循环是同步的（model.invoke / ToolNode 同步调用工具），
  而 MCP client 是 async 的（stdio + JSON-RPC 通信）。这里用 asyncio.run
  包一层，把「异步的 MCP 调用」变成「同步的 LangChain tool」。

通信方式（stdio）：
  每次调用启动一个 server 子进程（python mcp_servers/db_server.py），
  通过标准输入输出传 JSON-RPC，调用完就关闭。无状态、无长连接，最稳；
  对「查一次历史」这种低频调用，起一次进程的开销可接受。
"""
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langchain_core.tools import tool  # noqa: E402

SERVER_SCRIPT = os.path.join(ROOT, "mcp_servers", "db_server.py")


async def _call_mcp(tool_name: str, args: dict) -> str:
    """连一次 MCP server，调一个工具，把结果拼成文本返回。"""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # 用「当前 Python」去跑 server 脚本，保证两边用同一个解释器/环境；
    # env 显式传 os.environ.copy()：MCP 子进程默认不继承环境变量，
    # 不传的话 db.py 拿不到 MYSQL_URL，会退回 localhost 连不上 MySQL。
    params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
    parts = []
    for c in result.content:
        text = getattr(c, "text", None)
        parts.append(text if text is not None else str(c))
    return "\n".join(parts).strip()


@tool
def mcp_get_user_count() -> str:
    """统计系统里注册了多少用户（通过 MCP 标准协议连数据库 server 拿到）。"""
    return asyncio.run(_call_mcp("get_user_count", {}))


@tool
def mcp_get_history(user_id: int) -> str:
    """查某个用户的最近分析历史。user_id 是用户编号（通过 MCP 协议连数据库拿到）。"""
    return asyncio.run(_call_mcp("get_history_by_user", {"user_id": user_id}))


MCP_TOOLS = [mcp_get_user_count, mcp_get_history]
