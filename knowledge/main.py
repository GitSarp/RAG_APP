from zhipuai_llm import ZhipuAILLM
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from zhipuai_embedding import ZhipuAIEmbeddings
import os

_ = load_dotenv(find_dotenv())
api_key = os.environ["ZHIPUAI_API_KEY"]

file_paths = []
folder_path = './docs'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])

loaders = []
for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
texts = []
for loader in loaders: texts.extend(loader.load())
#todo 数据清洗

#text = texts[1]
#print(f"每一个元素的类型：{type(text)}.", 
#    f"该文档的描述性数据：{text.metadata}", 
#    f"查看该文档的内容:\n{text.page_content[0:]}", 
#    sep="\n------\n")


# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(texts)

#embedding = ZhipuAIEmbeddings()
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    #在线下载
    #model_name="moka-ai/m3e-base",
    model_name="./m3e-base",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为文本生成向量表示用于文本检索"
)
persist_directory = './vector_db/chroma'
#!rm -rf './vector_db/chroma'  # 删除旧的数据库文件（如果文件夹中有文件的话）
vectordb = Chroma.from_documents(
    #documents=split_docs[:20], # 为了速度，只选择前 20 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
vectordb.persist()
#print(f"向量库中存储的数量：{vectordb._collection.count()}")

# 加载数据库
#embedding = ZhipuAIEmbeddings()
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    #在线下载
    #model_name="moka-ai/m3e-base",
    model_name="./m3e-base",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为文本生成向量表示用于文本检索"
)
persist_directory = './vector_db/chroma'
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")

question = "40亿QQ号，1G内存，如何去重？"
docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(docs)}")
#for i, doc in enumerate(docs):
    #print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")


#提问大模型
llm = ZhipuAILLM(model="chatglm_std", temperature=0, api_key=api_key)
res = llm("请介绍下你自己！")
print(res)

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

question_1 = "40亿QQ号，1G内存，如何去重？"
print(question_1)
result = qa_chain({"query": question_1})
print("大模型+知识库后回答 question_1 的结果：")
print(result["result"])

question_2 = "什么是布隆过滤器？"
print(question_2)
result = qa_chain({"query": question_2})
print("大模型+知识库后回答 question_2 的结果：")
print(result["result"])


### 大模型自己的问答
prompt_template = """请回答下列问题:
                            {}""".format(question_1)
res = llm.predict(prompt_template)
print("大模型自己的回答 question_1 的结果：")
print(res)



