'''fastapi后端'''
# http://localhost:8501
#http://localhost:8000/docs

# ============ 第 1 块：导入库 + 创建应用 ============
#一次上传多次文件
from fastapi import FastAPI,Depends,Header,HTTPException,UploadFile,File
from typing import List

from pydantic import BaseModel
#FastAPI 的「安全工具箱」，鉴权相关的工具都从这拿。
#HTTPBearer —— 一个「自动验票闸机」类。它专门去 HTTP 请求头里找 Authorization: Bearer xxx
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
#HTTPAuthorizationCredentials —— 一个「数据盒子」类，
# 用来装闸机检查完的结果。盒子里有个字段 .credentials，装的就是纯 token。

from langgraph.types import Command
from omics_agent.graph import build_graph

import uuid    #专门造「全世界唯一」的编号
import os
import time
import jwt     # 签发 + 验证 JWT（登机牌）
import bcrypt  # 密码哈希（存密码用）
from dotenv import load_dotenv

#redis缓存库
import redis
import hashlib#算哈希

# 连接 Redis（全局一条连接，反复用）127.0.0.1 = 本机  + 端口+让取回来的值自动转成字符串
redis_client=redis.Redis(host=os.getenv("REDIS_HOST","127.0.0.1"),port=int(os.getenv("REDIS_PORT","6379")),decode_responses=True)

#存用户表到数据库 ====1.查用户 ===2.插入新用户
from db import get_user_by_username,create_user,delete_user,update_password,save_history,update_history_report,get_history_by_user


load_dotenv()
app=FastAPI()
  # 标准验牌方案：Swagger 会因此出现 Authorize 按钮；auto_error=False 让我们自己提示中文
graph=build_graph()#graph 放在外面（全局)服务是「常驻」的，造图成本高，造一次反复用



security = HTTPBearer(auto_error=False)#造闸机。auto_error=False —— 一个开关。默认闸机发现问题会自己抛 403（英文提示 Not authenticated）；
# 设成 False 就是「发现问题别自动报错，把结果交给我」，这样我们才能保留中文的「没带登记牌」

# ============ 第 2 块：JWT 配置常量 ============
SECRET_KEY = os.getenv("SECRET_KEY")# 签名用的私章，从 .env 读，别写死在代码里
ALGORITHM="HS256"                    # 签名算法（对称：一把钥匙既能签又能验）
ACCESS_TOKEN_EXPIRE_MINUTES=60*60# 登机牌有效期 1 小时


# ============ 第 3 块：密码哈希（bcrypt）============
def hash_password(password:str)->str:
    # hashpw(密码字节, 盐)：拿「密码 + 随机盐」算出哈希
    #盐 = 随机字符，作用：同一个密码每次哈希结果都不同，防止攻击者拿现成的「密码→哈希」对照表（彩虹表）撞库
    return bcrypt.hashpw(password.encode("utf-8"),bcrypt.gensalt()).decode("utf-8")

def verify_password(password:str,hashed:str)->bool:
    # 用「哈希里自带的那份盐」把用户输入的密码再算一遍，和原哈希比对，一致返回 True
    return bcrypt.checkpw(password.encode("utf-8"),hashed.encode("utf-8"))


# ============ 第 4 块：签发 token（发登机牌）============
def create_token(user_id:int,username:str)->str:
    payload={
        "sub":str(user_id), # sub = subject，JWT 标准字段，放「是谁」
        "username":username,
        "exp":int(time.time()+ACCESS_TOKEN_EXPIRE_MINUTES)## 过期时间戳
    }
    return jwt.encode(payload,SECRET_KEY ,algorithm=ALGORITHM)# 盖章成 token




# ============ 第 5 块：注册账号接口============
class RegisterRequest(BaseModel):
    username:str
    password:str

@app.post("/register")
def register(req:RegisterRequest):
    if get_user_by_username(req.username):        # 先查有没有同名
        raise HTTPException(status_code=400,detail="用户名已存在")
    hashed=hash_password(req.password)
    create_user(username=req.username,hashed_password=hashed)# 密码算哈希
    return{"message":"注册成功"}            # 存进库


# ============ 第 6 块：登录接口 === 查库===（发 token 的窗口）============
class LoginRequest(BaseModel):
    username:str
    password:str

@app.post("/login")
def login(req:LoginRequest):
    # ① 查用户存不存在
    user=get_user_by_username(req.username) # 查库
    # ② 用户不存在 或 密码不对 → 统一回「用户名或密码错误」
    #    故意不区分是哪个错，防止别人一个个试出「哪些用户名存在」
    #"把你输入的密码再按一次指纹，跟库里存的指纹对一下
    if user is None or not verify_password(req.password,user.hashed_password):#输入的密码, 库里的指纹
        raise HTTPException(status_code=401,detail="用户名或密码错误")
    # ③ 都对了 → 现场签一张登机牌发出去
    token=create_token(user.id,user.username)
    return {"access_token":token,"token_type":"Bearer"}
    #前端既需要牌（access_token），也需要知道怎么用它（token_type 决定放哪、加什么前缀）


# ============ 第 7 块：验签（安检验牌）============
# ① 依赖函数：就是"门口的验票员"，负责检查 token，通过则返回身份
#authorization从 HTTP 请求头（Header）里取，翻到了用翻到的值，翻不到才给 None
def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    # HTTPBearer 已经帮你检查了"有没有带 Bearer 前缀"，并摘掉前缀
    # 所以 credentials 是 None 就代表"没带牌或格式不对"
    if credentials is None:# 闸机说"没带牌"（因为 auto_error=False，它不报错，只给 None）
        raise HTTPException(status_code=401, detail="没带登记牌")

    # credentials.credentials 就是摘掉 "Bearer " 后的纯 token，直接拿去验
    token = credentials.credentials
    try:#下面这行可能抛异常，包起来准备接。
        #验签通过 → 返回解出来的 payload 字典（就是我们 encode 时塞进去的那些）
        payload=jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:# 过期（它是 InvalidTokenError 的子类，要先捕获）
        raise HTTPException(status_code=401, detail="登机牌过期了")
    except jwt.InvalidTokenError: # 被篡改 / 格式错
        raise HTTPException(status_code=401, detail="登机牌是假的或被动过")
    # 验票通过，返回「身份牌」
    return {"user_id":int(payload["sub"]),"username":payload["username"]} # 验票通过，给"身份牌"

# # ② /me：验票的试金石，返回「我是谁」
@app.get("/me")
def read_me(user:dict=Depends(get_current_user)):
    return user# 这里拿到的 user 已经是"验过票"的身份，不用再自己校验


# ============ 第 8 块：删除账号接口============
@app.delete("/user")#
def delete_account(user:dict=Depends(get_current_user)):
#删除要「先验票」——只有登录的人才能删自己，没登录的人碰不到这个接口
    delete_user(user["username"])
    return {"message":"账号已删除"}


# ============ 第 9  块：改变账号密码接口============
class ChangePasswordRequest(BaseModel):
    old_password:str
    new_password:str

@app.post("/change_password")
def change_password(req:ChangePasswordRequest,user:dict=Depends(get_current_user)):
     # 先查库拿完整用户（验旧密码要用 hashed_password）
     do_user=get_user_by_username(user["username"])
     if not verify_password(req.old_password,do_user.hashed_password):# 验旧密码
         raise HTTPException(status_code=401,detail="旧密码不对")
     update_password(user["username"],hash_password(req.new_password))# 改密码落库
     return  {"message":"密码修改成功"}

# ============ 第 10 块：业务接口（/analyze + /confirm）============
#上传多个文件
@app.post("/upload")
# List[UploadFile] 类型标注：一个列表，里面每个都是UploadFile（上传文件）
def upload_files(files:List[UploadFile]=File(),user:dict=Depends(get_current_user)):
    #存"这次任务的目录路径"
    task_dir=os.path.join("data","upload",str(uuid.uuid4()))
    #数据根目录 + 上传文件放这 +每次来都发一个不重复的门牌号

    os.makedirs(task_dir,exist_ok=True)#把这个目录建出来；已经有了就跳过

    for f in files:
        #把当前文件的内容全读出来，存进 content
        content=f.file.read()
        #b = 二进制（binary，因为文件是字节不是文字）
        with open(os.path.join(task_dir,f.filename),"wb") as out:
            out.write(content)

    #把读到的字节写进新文件里（完成"存盘"）
    return{"dataset_path":task_dir}

class AnalizeRequest(BaseModel):
    user_request:str
    dataset_path:str='data'
    user_id:str="request-01"

class ConfirmRequest(BaseModel):
    thread_id:str
    approved:bool


#Depends 隐形一步：先自动调 get_current_user（摘前缀 + 验签）
#身份令牌（Authorization: Bearer <token>）
# FastAPI从请求头抽 Authorization 喂给 get_current_user

#声明中断+人工审核接口
@app.post('/analyze')#返回分析结果给前端
def analyze(asg:AnalizeRequest,user:list=Depends(get_current_user)):

    # ① 先查缓存：这个请求之前算过吗？将用户问题MD5计算结果存成key,等下查重
    cache_key="report:"+hashlib.md5(asg.user_request.encode("utf-8")).hexdigest()
    cached = redis_client.get(cache_key)# 用连接对象调 get，按 key 去 Redis 里取值
    if cached:  # 之前问过完全相同的问题+返回告前端"完成，直接显示"+缓存里的报告 +标记，"缓存命中"
        return {"status": "done", "report": cached, "from_cache": True}

    thread_id=str(uuid.uuid4())
    config={"configurable":{"thread_id":thread_id}}
    initial={
        "user_request":asg.user_request,
        "dataset_path":asg.dataset_path,
        "user_id":asg.user_id
    }
    # 第一次 invoke，checkpointer 用这个编号存检查点
    result=graph.invoke(initial,config=config)#整条流水线跑完（或暂停时）的"状态背包
    if result.get("__interrupt__"):
        info=result["__interrupt__"][0].value
        # 质检需人工确认：先存一条「待确认」（报告还没出）
        save_history(user_id=user["user_id"],thread_id=thread_id,
                     user_request=asg.user_request,status="pending")
        return {
            "status":"pending_confirm",#（报告还没出，先记个"进行中"）
            "thread_id":thread_id,
            "qc_report":info.get("qc_report"),
            "group_a":info.get("group_a"),
            "group_b":info.get("group_b"),
               }
    #  ② 直接出报告（质检不合格/类型不支持）：先取报告，再存缓存
    final_report = result.get("final_report")

    #调 set，往 Redis 里存一个 key-value
    redis_client.set(cache_key,final_report,ex=24*3600)#"24 小时后自动删"

    save_history(user_id=user["user_id"],thread_id=thread_id,
                 user_request=asg.user_request,status="done",final_report=final_report)
    return{"status":"done","report":final_report}
    #status: "done"：告诉前端：「已经跑完，直接显示报告，不用确认」信号


#声明中断结束分析生成最终报告
@app.post("/confirm")
def confirm(req:ConfirmRequest,user:list=Depends(get_current_user)):
    config={"configurable":{"thread_id":req.thread_id}}
    #第二次 invoke，checkpointer 用这个编号找检查点
    result=graph.invoke(Command(resume={"approved":req.approved}),config=config)
    while result.get("__interrupt__"):
        #只要还中断，就一直续跑
        result=graph.invoke(Command(resume={"approved":req.approved}),config=config)
    final_report=result.get("final_report")
    # 报告出全了：按 thread_id 找到当初那条 pending，补报告 + 改 done
    update_history_report(thread_id=req.thread_id,final_report=final_report)
    return {"report":final_report}


#声明一个"查历史"的 GET
@app.get("/history")
def get_history(user:dict=Depends(get_current_user)):
    histories=get_history_by_user(user["user_id"])  #  查出这个用户的所有历史（对象列表）
    # 对象 → 字典，前端才能收 JSON
    return[
        {"id":h.id,"user_request":h.user_request,"final_report":h.final_report,
    "status":h.status,"created_at":h.created_at}
        for h in histories#把 histories 里每个元素，挨个套进 {...} 这个模板变成新东西
    ]