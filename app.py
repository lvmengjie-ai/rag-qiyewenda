import gradio as gr

# ===================== ModelScope阿里国内永久部署全兼容 =====================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatTongyi
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# ===================== 你的通义千问阿里云API密钥 =====================
api_key = "sk-f1f1cdcad2bc498c8683996a1ee4987b"
llm = ChatTongyi(
    model_name="qwen-turbo",
    dashscope_api_key=api_key,
    temperature=0.1
)

# ===================== 企业内部知识库 =====================
knowledge = """
公司考勤制度：
1. 上班时间：上午9:00，下班时间：下午18:00
2. 午休时间：12:00-13:00
3. 迟到15分钟内不计考勤

请假制度：
1. 事假需提前1天申请
2. 病假需提供医院证明
3. 试用期无带薪年假

福利制度：
1. 入职缴纳五险一金
2. 满一年5天带薪年假
"""

# ===================== 文档切片 =====================
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
splits = splitter.split_text(knowledge)

# ===================== 向量库（全程无外网依赖） =====================
embeddings = FakeEmbeddings(size=384)
db = FAISS.from_texts(splits, embeddings)

# ===================== BM25+向量混合检索 =====================
vec_ret = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})
bm25_ret = BM25Retriever.from_texts(splits)
bm25_ret.k = 2
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_ret, vec_ret],
    weights=[0.4, 0.6]
)

final_retriever = hybrid_retriever

# ===================== 幻觉约束提示词 =====================
prompt = PromptTemplate.from_template("""
你是企业内部智能助手，仅根据下方参考资料回答。
资料未提及的内容，统一回复：暂无相关制度信息。
禁止编造、禁止扩展，回答简洁正式。

参考资料：{context}
员工问题：{question}
""")

# ===================== RAG问答链 =====================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=final_retriever,
    chain_type_kwargs={"prompt": prompt}
)

# ===================== Gradio聊天界面 =====================
def chat_fn(msg, history):
    result = qa_chain.invoke({"query": msg})
    return result["result"]

demo = gr.ChatInterface(
    fn=chat_fn,
    title="企业内部管理制度RAG知识库系统",
    description="阿里云通义千问 · 本地知识库智能问答"
)

# ===================== 平台官方启动参数 =====================
if __name__ == "__main__":
    demo.launch()
