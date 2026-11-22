"""MCP 客户端桥接：把 MCP server 的工具「转成」LangChain 同步工具。

为什么要有这个文件：
  agent.py 的 ReAct 循环是同步的（model.invoke、工具调用都是同步函数），
  而 MCP client 是 async 的（stdio + JSON-RPC 通信）。这里负责把
  「异步的 MCP 调用」变成「同步的 LangChain tool」。

改之前的两处硬伤（面试可以直接讲这两点）：
  1) 每次工具调用都新起一个 stdio 子进程 + 重新 initialize 握手。
     一次调用至少几百 ms 的进程启动开销，多工具串起来就是秒级延迟，
     而且每次都要重新建立 stdio 管道和会话状态。
     → 现在改成「常驻会话」：子进程只起一次，之后所有调用复用同一条
       stdio 连接（MCP 的 JSON-RPC 自带 request id，本来就支持复用连接）。
  2) 用 asyncio.run() 把异步包成同步。这玩意儿只能在「当前线程没有运行中
     事件循环」时用；一旦这个工具被 FastAPI 的 async 路由（或任何已跑在
     event loop 里的上下文）调用，立刻抛
     RuntimeError: asyncio.run() cannot be called from a running event loop。
     → 现在改成「后台线程 + 常驻事件循环」：会话和它的事件循环永远活在那条
       后台线程里；同步调用方只做两件事——把请求投进线程安全的队列、
       等一个 concurrent.futures.Future。调用方从头到尾不碰 asyncio，
       所以无论被同步代码还是 async 路由调用都不会冲突。

  顺带解决第三个问题：单次工具调用有超时上限，server 卡死/慢查询不会把
  整个 ReAct 请求拖到超时；超时只回收「这条连接」，事件循环线程继续活着，
  后续请求在新连接上照常执行（这里自测抓出过一个坑，见 _serve 的注释）。

通信方式（stdio）：一个常驻子进程（python mcp_servers/db_server.py），
  通过标准输入输出传 JSON-RPC。
"""
import asyncio
import atexit
import concurrent.futures
import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langchain_core.tools import tool  # noqa: E402

SERVER_SCRIPT = os.path.join(ROOT, "mcp_servers", "db_server.py")

# 单次工具调用超时（秒）。默认 15s：比 agent.py 里单工具 30s 的上限更短，
# 目的是让 MCP 这一层先给出「我超时了」的友好文案，而不是被外层一刀切掉。
CALL_TIMEOUT = float(os.getenv("MCP_CALL_TIMEOUT", "15"))
# 建会话（起子进程 + initialize 握手）的超时（秒）
STARTUP_TIMEOUT = float(os.getenv("MCP_STARTUP_TIMEOUT", "20"))
# 连接被回收后重连前的等待（秒），避免 server 一崩就疯狂重试打满 CPU
RECONNECT_DELAY = float(os.getenv("MCP_RECONNECT_DELAY", "0.2"))


def _render_result(result) -> str:
    """把 MCP 的 CallToolResult 拍平成一段文本（和改之前的行为保持一致）。"""
    parts = []
    for content in getattr(result, "content", None) or []:
        text = getattr(content, "text", None)
        parts.append(text if text is not None else str(content))
    return "\n".join(parts).strip()


class McpSession:
    """常驻 MCP 会话：后台线程 + 常驻事件循环 + 长期存活的 stdio 子进程。

    线程模型（这是整个文件的核心，也是面试时最值得讲的部分）：

        调用方线程（可能是 FastAPI worker、也可能是 LangGraph 的线程池）
            │  call()：投一个 (future, 工具名, 参数, 超时) 进队列
            ▼
        ┌──────────── 后台线程（name=mcp-session，常驻）────────────┐
        │  asyncio 事件循环（只在这一条线程里跑）                   │
        │    └─ _serve：断线重连的外层循环                          │
        │         └─ _serve_connection：建连 + 从队列取请求执行     │
        └──────────────────────────────────────────────────────────┘

    两个关键设计取舍：
      · 子进程/连接只建一次（省掉每次重连的开销）；
      · 调用方不持有事件循环 → 永远不会触发
        「asyncio.run() cannot be called from a running event loop」。
    另外「事件循环线程」与「连接」是两层，可以分开生老病死：
    调用超时只回收连接（外层循环立刻重连），事件循环线程和队列都不动。
    这样排队中的请求只是多等一次握手，不会因为一次超时就整条链路失效。
    """

    def __init__(self, server_script: str = SERVER_SCRIPT,
                 call_timeout: float = CALL_TIMEOUT,
                 startup_timeout: float = STARTUP_TIMEOUT) -> None:
        self._server_script = server_script
        self._call_timeout = call_timeout
        self._startup_timeout = startup_timeout
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        # 「循环已经在跑」的信号：由循环自己置位（见 _mark_serving），
        # 所以置位就等于「此刻投递不会丢」。
        self._ready = threading.Event()
        # 「循环还在服务」的信号：线程退出前第一时间清掉。派发前看一眼，
        # 避免把请求投给一个已经开始退出的循环（投进去就是石沉大海）。
        self._serving = threading.Event()
        # 「这个会话被要求停止了」：close() 或卡死后重建时置位
        self._stopping = threading.Event()
        self._start_lock = threading.Lock()   # 只保护「启动后台线程」这一步
        self._last_error: str | None = None

    # ══════════ 对外：同步调用 ══════════
    def call(self, tool_name: str, args: dict, timeout: float | None = None) -> str:
        """同步调用一个 MCP 工具。线程安全，可以在任意线程/事件循环里调用。"""
        per_call = timeout or self._call_timeout
        if not self._ensure_started():
            return f"MCP 会话启动失败：{self._last_error or '未知原因'}"
        loop, queue = self._loop, self._queue
        if loop is None or queue is None:
            return f"MCP 会话不可用：{self._last_error or '事件循环已退出'}"

        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            # 关键一步：入队这个动作被「搬」到会话线程里执行。
            # 用 call_soon_threadsafe 而不是 asyncio.run/create_task，是因为
            # 调用方这边根本不需要 await——它只要一个能被阻塞等待的 Future。
            loop.call_soon_threadsafe(queue.put_nowait, (future, tool_name, args, per_call))
        except RuntimeError as exc:   # 事件循环已经关掉了（close() 之后）
            return f"MCP 会话已关闭：{exc}"
        try:
            # 硬上限比内部超时多留 5s：内部超时负责给出友好文案，
            # 这里只防「事件循环线程自己卡死」——那种情况下面已经发不出结果了。
            return future.result(timeout=per_call + 5.0)
        except concurrent.futures.TimeoutError:
            # 能走到这里，说明「连内部超时都没能发出来」——卡住的是事件循环线程
            # 本身，不是那个工具。把整个会话标记为不可用，下次 call() 重建，
            # 而不是继续往同一堵墙上撞。
            self._invalidate()
            return (f"MCP 调用超时：{tool_name} 超过 {per_call:g} 秒未返回，"
                    "本次调用已放弃（会话会在下一次调用时重建）。")

    def close(self) -> None:
        """关掉常驻会话：投一个关闭信号，让会话线程自己退出（子进程随之结束）。"""
        self._stopping.set()
        loop, queue = self._loop, self._queue
        if loop is None or queue is None:
            return
        try:
            # 用哨兵把可能正卡在 queue.get() 上的循环叫醒
            loop.call_soon_threadsafe(queue.put_nowait, (None, None, None, 0))
        except RuntimeError:
            pass   # 循环已经关了，那就没什么要关的了

    # ══════════ 内部：线程与事件循环 ══════════
    def _ensure_started(self) -> bool:
        """确保后台线程与事件循环在跑；线程挂了（上次会话崩了）就重建。"""
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                if self._serving.is_set() and not self._stopping.is_set():
                    return True
                # 线程还活着但已经不再服务（正在退出）：等它退干净再重建。
                # 不等就会把请求投给一个跑完/正在关的循环——调用方会一直等到
                # 硬超时才拿到「超时」，表现得像「工具慢」，排查时极具误导性。
                self._thread.join(timeout=self._startup_timeout)
            self._ready.clear()
            self._serving.clear()
            self._stopping.clear()   # 新会话重新接受请求
            self._last_error = None
            self._thread = threading.Thread(
                target=self._thread_main, name="mcp-session", daemon=True
            )
            self._thread.start()
            if not self._ready.wait(self._startup_timeout):
                self._last_error = f"事件循环 {self._startup_timeout:g}s 内没有就绪"
                return False
            return True

    def _thread_main(self) -> None:
        """后台线程入口：建事件循环，跑会话协程，退出时把没回音的结果兜掉。"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._queue = asyncio.Queue()
        # 用 call_soon 而不是直接 set：这个回调要等 loop 真正跑起来才执行，
        # 所以「_ready 被 set」等价于「循环已在运行」，调用方此时投递不会丢。
        loop.call_soon(self._mark_serving)
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:   # noqa: BLE001 - 会话线程不能把异常甩给整个进程
            self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            # 先把「在服务」的灯关掉，再清理队列：顺序反过来的话，中途进来的
            # 请求会看到灯还亮着，然后被投进一个已经没人消费的队列。
            self._serving.clear()
            # 线程退出前把还在排队的请求全部回一条「会话挂了」，
            # 免得调用方一直在 future.result() 上等到硬超时。
            self._fail_pending(self._last_error or "会话已结束")
            self._loop = None
            self._queue = None
            loop.close()

    def _mark_serving(self) -> None:
        """由事件循环自己在启动时调用：点亮「在服务」的灯，再放行调用方。"""
        self._serving.set()
        self._ready.set()

    def _invalidate(self) -> None:
        """把一个卡死的会话标记为不可用（不 join，避免调用方被一起拖住）。

        只丢引用不等待：旧线程可能永远醒不过来，join 只会把调用方也卡住。
        代价是最多残留一个卡死的线程，这比「整条请求链路陪葬」划算得多。
        """
        self._stopping.set()
        self._serving.clear()
        self._thread = None
        self._loop = None
        self._queue = None

    async def _serve(self) -> None:
        """外层循环：只负责「重连」，不负责「退出」。

        为什么要分成内外两层（这是自测抓出来的一个真 bug）：
          最初写成「一次超时就 return，整个协程结束」——于是事件循环跟随连接
          一起结束，线程进入退出流程。此时如果调用方已经拿到超时文案、紧接着
          发起下一次调用，请求就会被投进这条正在退出的循环，再也没有人消费，
          白等到硬超时（表现为「第二个调用也超时，连接数还是 1」）。
          现在把「连接」和「事件循环线程」解耦：超时只回收连接，本协程
          立刻重建一条新连接继续消费同一个队列，排队中的请求一个都不丢。
        """
        while not self._stopping.is_set():
            try:
                await self._serve_connection()
            except asyncio.CancelledError:
                raise
            except Exception as exc:   # noqa: BLE001 - 建连失败：报错并重试
                self._last_error = f"{type(exc).__name__}: {exc}"
                # 建连就失败了：把排队里的请求先回掉（含 ImportError 这种
                # 装包没装好的情况），别让调用方空等到超时才发现原因。
                self._fail_pending(f"连接 MCP server 失败：{exc}")
                await asyncio.sleep(RECONNECT_DELAY)

    async def _serve_connection(self) -> None:
        """建一条连接并从队列里取请求执行；返回即表示「该重连了」。"""
        # 惰性 import：模块级 import mcp 会让「没装 mcp 时连 import 都失败」，
        # 而这里延迟到真正要连 server 的时候，报错也能变成一句可读的文案。
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # 用「当前 Python」去跑 server 脚本，保证两边同一个解释器/环境；
        # env 显式传 os.environ.copy()：MCP 子进程默认不继承环境变量，
        # 不传的话 db.py 拿不到 MYSQL_URL，会退回 localhost 连不上 MySQL。
        params = StdioServerParameters(
            command=sys.executable,
            args=[self._server_script],
            env=os.environ.copy(),
        )
        queue = self._queue
        # stdio_client / ClientSession 内部用 anyio 的 task group，必须在
        # 「同一个 task」里进出（否则报 cancel scope 跨 task 的错）。所以
        # 建连和 get→call_tool 的循环都写在这一个协程里，不交给别的 task。
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), self._startup_timeout)
                while not self._stopping.is_set():
                    future, tool_name, args, per_call = await queue.get()
                    if future is None:       # close() 投的关闭信号
                        self._stopping.set()
                        return
                    try:
                        result = await asyncio.wait_for(
                            session.call_tool(tool_name, args), per_call
                        )
                    except asyncio.TimeoutError:
                        future.set_result(
                            f"MCP 调用超时：{tool_name} 超过 {per_call:g} 秒未返回，"
                            "本次调用已放弃。"
                        )
                        # 超时后这条连接的内部状态不可信（请求可能还挂在对端，
                        # 响应迟早会乱序回来），回收它；外层循环会重连，
                        # 队列里其它请求改在新连接上执行，一条都不丢。
                        self._last_error = f"调用 {tool_name} 超时，连接已回收"
                        return
                    except Exception as exc:   # noqa: BLE001 - 单次失败不该拖垮会话
                        future.set_result(f"MCP 工具 {tool_name} 调用失败：{exc}")
                    else:
                        future.set_result(_render_result(result))

    def _fail_pending(self, reason: str) -> None:
        """把队列里还没被处理的请求逐个回一条错误，避免调用方空等到超时。"""
        queue = self._queue
        if queue is None:
            return
        while True:
            try:
                future, _name, _args, _timeout = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            except Exception:   # noqa: BLE001 - 队列已跟着 loop 一起废了
                return
            if future is not None and not future.done():
                future.set_result(f"MCP 会话异常：{reason}")


# 模块级单例：整个进程一个常驻会话（子进程只起一次）。
_SESSION = McpSession()


@tool
def mcp_get_user_count() -> str:
    """统计系统里注册了多少用户（通过 MCP 标准协议连数据库 server 拿到）。"""
    return _SESSION.call("get_user_count", {})


@tool
def mcp_get_history(user_id: int) -> str:
    """查某个用户的最近分析历史。user_id 是用户编号（通过 MCP 协议连数据库拿到）。"""
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        # 模型偶尔会把用户 ID 写成 "user-3" 这种，先在工具里挡掉再让它自己纠。
        return f"user_id 必须是整数编号，收到的是：{user_id!r}"
    return _SESSION.call("get_history_by_user", {"user_id": uid})


MCP_TOOLS = [mcp_get_user_count, mcp_get_history]


def close_mcp_session() -> None:
    """显式关闭入口（测试收尾、服务停机时调用；进程退出由 atexit 兜底）。"""
    _SESSION.close()


# 进程退出时把子进程收干净，不然 dev 环境里会攒一堆孤儿 python 进程
atexit.register(_SESSION.close)
