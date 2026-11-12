'''streamlit前端'''

import streamlit as st
import requests#专门用来发 HTTP 请求（调你的 FastAPI 接口）
import os

# ======1.FastAPI 的地址=====
BASE=os.getenv("BACKEND_URL","http://127.0.0.1:8000")

#=====工具一：一个「自动帮你刷卡的小助手」不用每处重写。======
#从 session_state 里掏出 token，拼成「带卡」的请求头；没登录就返回空头
def get_header():
    token=st.session_state.get("token")
    return {"Authorization":f"Bearer {token}"} if  token else {}
#把「掏 token + 拼请求头」这个重复动作抽出来，三处调接口都能用，

#=====工具二：登录开关=====
if "login_in" not in st.session_state:#登录开关：一开始是「关」（没登录）
    st.session_state["login_in"]=False


#========2.分支一：没登录 → 2.分支一：没登录 → 显示「登录 / 注册」两个模式= |   jwt: 给前端加"登录"(先拿卡)==========
if not st.session_state["login_in"]:
    st.title("多组学数据分析 Agent")

    # 用一个单选切换「登录」还是「注册」，用户选哪个，mode 就是哪个
    mode=st.radio("选择操作",["登录","注册"])

    # 用户名、密码输入框（登录和注册共用这一对框，切模式内容还在）
    username=st.text_input("用户名")
    password=st.text_input("密码",type="password")# type="password" 让密码显示成 ●●●●

    # —— 模式一：登录 ——
    if mode=="登录":
      if st.button("登录"):
          resp=requests.post(f"{BASE}/login",json={"username":username,"password":password})
          if resp.status_code==200:
              st.session_state["token"]=resp.json()["access_token"]# 拿到卡，揣进兜里
              st.session_state["login_in"]=True
              st.success("登录成功")
          else:
              st.error("用户名或密码错误")
       # Streamlit 的提示框：success 绿色成功、error 红色报错

    # —— 模式二：注册 ——
    else:
        if st.button("注册"):
            resp=requests.post(f"{BASE}/register",json={"username":username,"password":password})
            if resp.status_code==200:
                st.success("注册成功")
            else:
               st.error(resp.json().get("detail","注册失败"))
               # get("detail","注册失败") 指 取 detail 字段的值（就是"用户名已存在"）
               # 万一后端没返回 detail，就兜底显示"注册失败"


#========分支二：已登录 → 显示 Agent 分析界面  |   3.jwt:让前端"带卡进门"（每次调接口都带 token）=====
else:
    st.title("多组学数据分析 Agent")

# ===== 账号管理（登录后才有）=====
    #st.expander折叠面板：页面上显示一个「账号管理」标题，点一下展开、再点收起
    with st.expander("账号管理"):
        # —— 改密码 ——
        #st.subheader() 三级小标题
        st.subheader("改密码")
        old_pwd=st.text_input("旧密码",type="password")
        new_pwd = st.text_input("新密码", type="password")
        if st.button("确认修改"):
            resp=requests.post(f"{BASE}/change_password",json={"old_password":old_pwd,"new_password":new_pwd},headers=get_header())
            if resp.status_code==200:
                st.success("密码修改成功")
            else:
                st.error(resp.json().get("detail","修改失败"))

        # —— 删除账号 ——
        st.subheader("删除账号")
        if st.button("⚠️ 删除我的账号"):
            resp=requests.delete(f"{BASE}/user",headers=get_header())
            if resp.status_code==200:
                st.session_state["login_in"]=False  # 账号没了，退出
                st.session_state.pop("token",None)
            else:
                st.error("删除失败")

        # ===== 历史记录（登录后才有）=====
        with st.expander("历史记录"):
            if st.button("刷新纪录"):
                resp=requests.get(f"{BASE}/history",headers=get_header())
                if resp.status_code==200:
                    histories=resp.json()
                    if not histories:
                        st.write("暂无历史记录")
                    for h in histories:
                        status_label="✅ 完成" if h["status"]=="done" else "⌛待确认"
                        with st.expander(f"{status_label} | {h['created_at']} | {h['user_request'][:20]}"):
                            st.write(h["final_report"])
                else:
                    st.error("获取历史失败")


#=====退出登录=====
    if st.button("退出登录"):
        st.session_state["login_in"]=False      # 开关关掉
        st.session_state.pop("token",None)      # 清掉 token（pop 第二个参数=键不存在也不报错）
        st.rerun()                              #重新演一遍，回到登录框

    # ===== 上传数据文件（先传数据，再分析）=====
    #accept_multiple_files=True —— 允许一次选多个文件====uploaded存"用户选的文件
    uploaded = st.file_uploader("上传数据文件(csv)",accept_multiple_files=True)
    if uploaded and st.button("上传数据"):
        #("files",(f.name,f.getvalue()))一个元组 (字段名，必须和后端参数名一致, (文件的原名,  这个文件的字节内容))
        files=[("files",(f.name,f.getvalue())) for f in uploaded]
        # files=files： 用表单文件的方式发（不是 json），发的是上面打包好的列表
        resp=requests.post(f"{BASE}/upload",files=files,headers=get_header())
        if resp.status_code==200:
            #把后端返回的"目录路径"存进 session_state，供后面 /analyze 用。
            # resp.json() 把服务器的返回转成字典
            st.session_state["dataset_path"]=resp.json()["dataset_path"]
            st.success(f"上传成功: {st.session_state['dataset_path']}")
        else:
            st.error("上传失败")#st.error(...)  画一个红色报错提示框
    # ── 引导：告诉用户这个 Agent 支持 4 种用法 ──
    st.caption(
        "支持 4 种用法：①差异分析「帮我找差异表达基因」 ②PCA「做主成分分析」 "
        "③聚类「做聚类分析」 ④知识问答「哪些基因参与 DNA 损伤修复？」"
    )
    # st.caption(文字) = Streamlit 显示一行灰色小字，专门用来做提示/说明

    user_request=st.text_input("你的分析需求:","帮我找差异表达基因并解读")

    if st.button("开始分析"):
        # ── 先清空上一次分析留下的状态，防止新旧结果混着显示 ──
        # 三个键分别是：待确认的 thread_id、质检报告、直接出结果的报告
        st.session_state["thread_id"] = None
        st.session_state["qc_report"] = None
        st.session_state["done_report"] = None

        # 它装的可能是「问答 / PCA / 聚类 / 质检失败」任何一份报告，不只是失败

        resp=requests.post(f"{BASE}/analyze",json={"user_request":user_request,
        "dataset_path":st.session_state.get("dataset_path","data")},headers=get_header())
        #get(键, 默认值)：盒子里有 dataset_path就用它（上传过）；没有就用默认"data"（没上传）

        if resp.status_code==401:
            st.error("请先登录再分析")
        else:
              data=resp.json()
              if data["status"]=="pending_confirm":
                  st.session_state["thread_id"]=data["thread_id"]
                  st.session_state['qc_report']=data.get("qc_report")
              else:
                   st.session_state["done_report"]=data.get("report")


    # ======= 情况1：直接出结果（质检失败 / 知识库问答 / PCA / 聚类 / 不支持）=========
    if st.session_state.get("done_report"):
        st.write("### 分析结果")
        st.write(st.session_state['done_report'])

    # ========情况2：质检合格=========
    if st.session_state.get("qc_report"):
        st.write("### 质检报告")
        st.write(st.session_state['qc_report'])
    if st.session_state.get("thread_id"):  # 有 thread_id（确实在等批准）才显示按钮
        if st.button("✅ 批准继续"):
            resp = requests.post(f"{BASE}/confirm", json={"thread_id": st.session_state["thread_id"], "approved": True},
                                 headers=get_header())
            st.write("### 最终报告")
            st.write(resp.json()["report"])
        if st.button("❌ 取消"):
            resp = requests.post(f"{BASE}/confirm",
                                 json={"thread_id": st.session_state["thread_id"], "approved": False},
                                 headers=get_header())
            st.write("### 分析已取消 ")
            st.write(resp.json()["report"])
