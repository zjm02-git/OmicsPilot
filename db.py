'''db.py：数据库连接 + 用户表模型 + 用户对话历史记忆（web.py 从这里拿用户.历史记忆）'''

# ============ 第 1 块：导入工具 ============
# create_engine：造「电话线」（连接数据库的管道）
# Column：画表格设计图时，用来画「一列」
# Integer / String：列的类型（整数 / 字符串）
from sqlalchemy import create_engine,Column,Integer,String

# declarative_base：造「图纸底板」，所有表的模型都继承它
# Session：造「话筒」，增删改查都通过它
from sqlalchemy.orm import declarative_base,Session

from datetime import datetime
import os

# ============ 第 2 块：连库（电话线）============
# 连接串 = 一条完整地址：用什么驱动连 + 账号密码 + 地址端口 + 连哪个库
# 真实项目密码别写死在这，要放 .env（这里 123456 只是本地演示）
DATABASE_URL=os.getenv("MYSQL_URL","mysql+pymysql://root:123456@localhost:3306/omics")
engine=create_engine(DATABASE_URL)# 照着地址，造一条通往 omics 库的管道


# ============ 第 3 块：模型（设计图）============
Base=declarative_base()  # 造一块「图纸底板」

#SQLAlchemy 内部有个"元类"（metaclass）.在定义 User 的时候，自动把 User 的设计图
# (表名 users + 三列）登记进 Base.metadata 这个登记册。
class User(Base):        # User 表的设计图，挂在底板上
    __tablename__="users"#这张表在 MySQL 里叫 users

    id=Column(Integer,primary_key=True)      # 主键列：整数、自动递增、唯一
    username=Column(String(50),unique=True)  # 用户名列：字符串、不能重复
    hashed_password=Column(String(100))             # 密码哈希列：字符串


# ============ 第 4 块：查用户 ============
def get_user_by_username(username):
    """按用户名查用户：查到返回 User 对象，查不到返回 None"""
    with Session(engine) as session:#Session(engine)告诉话筒「你插到 engine 这条电话线上」
        return session.query(User).filter(User.username==username).first()
#调话筒的 query 方法，参数是 User（表）


# ============ 第 5 块：插入新用户 ============
def create_user(username:str,hashed_password:str)->User :
    with Session(engine) as session: # 拿起话筒
        new_user=User(username=username,hashed_password=hashed_password)# 造一个 User 对象
        session.add(new_user)   # 放进待提交托盘
        session.commit()        # 提交落库
        return new_user


# ============ 第 6 块：删除用户 ============
def delete_user(username:str)->None:
    with Session(engine) as session:
        user=session.query(User).filter(User.username==username).first()
        session.delete(user)
        session.commit()
        return user


# ============ 第 7 块：改密码 ============
def update_password(username:str,new_hashed_password:str):
    """改密码：查 → 改 → 提交（必须在同一个 session 里）"""
    with Session(engine) as session:# 拿起话筒，这次不松手
        user=session.query(User).filter(User.username==username).first()
        user.hashed_password=new_hashed_password # 改原件
        session.commit()# 提交，真落库
        return user


# ============ 第 8 块：对话历史表（一对多：一个用户 → 多条历史）============
class AnalysisHistory(Base):
    __tablename__="analysis_history"# 表名

    id=Column(Integer,primary_key=True)  # 历史主键：每条历史一个编号
    user_id=Column(Integer,index=True)   # 外键：属于哪个用户（= users.id）；加索引让"按用户查历史"走目录，不再全表扫
    thread_id=Column(String(100))        # 对应哪次分析（confirm 时用它定位更新）
    user_request=Column(String(500))     # 用户问了什么
    final_report=Column(String(10000))   # 最终报告（长文本）
    status=Column(String(20))            # pending 待确认 / done 完成
    created_at=Column(String(100))       # 记录时间


# ============ 第 9 块：存历史（增）============
def save_history(user_id,thread_id,user_request,status,final_report=""):
    with Session(engine) as session:
        h=AnalysisHistory(#造一张「小票」对象，照着表逐列填值
            user_id=user_id,
            thread_id=thread_id,
            user_request=user_request,
            status=status,
            final_report=final_report,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        session.add(h)
        session.commit()
        return h


# ============ 第 10 块：更新历史报告（改，按 thread_id 定位）============
def update_history_report(thread_id,final_report):
    with Session(engine) as session:#（定位到「那一次分析」）
        h=session.query(AnalysisHistory).filter(AnalysisHistory.thread_id==thread_id).first()
        if h:
            h.final_report=final_report
            h.status="done"
            session.commit()
        return h


# ============ 第 11 块：查某用户的历史（查，倒序）============
#.order_by(AnalysisHistory.id.desc()) : │ 按 id 倒序：id 越大越新，desc() 让新的排最前
def get_history_by_user(user_id):# 参数：查谁的历史
    with Session(engine) as session:#desc()（descending 倒序的缩写）
        return session.query(AnalysisHistory).filter(AnalysisHistory.user_id==user_id).order_by(AnalysisHistory.id.desc()).all()


# ============ 第 12 块：建表 ============
Base.metadata.create_all(engine)  # 登记册」叫 metadata.照设计图，把 users 表,AnalysisHistory表在 MySQL 里真造出来