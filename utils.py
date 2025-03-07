from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain

from langchain_community.embeddings import DashScopeEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
# def generate_script(subject,video_length,creativity,api_key):
#     title_template=ChatPromptTemplate.from_messages(
#         [
#             ("human","请为’{subject}‘这个主题的视频想一个吸引人的视频脚本")
#         ]
#     )
#     script_template=ChatPromptTemplate.from_messages(
#         [
#             ("human","""你是一位短视频频道的博主。根据以下标题和相关信息，为短视频频道写一个视频脚本。
#              视频标题：{title}，视频时长：{duration}分钟，生成的脚本的长度尽量遵循视频时长的要求。
#              要求开头抓住限球，中间提供干货内容，结尾有惊喜，脚本格式也请按照【开头、中间，结尾】分隔。
#              整体内容的表达方式要尽量轻松有趣，吸引年轻人。""")
#         ]
#     )
#
#     model=ChatOpenAI(model="deepseek-chat",api_key=api_key,temperature=creativity,base_url="https://api.deepseek.com")
#
#     title_chain=title_template|model
#     script_chain=script_template|model
#
#     title=title_chain.invoke({"subject":subject}).content
#     script=script_chain.invoke({"title":title,"duration":video_length}).content
#     return title,script
#
#
# print(generate_script("deepseek模型",1,1,"sk-75fbcfe1039645dbb3d361962cbd82df"))


def qa_agent(api_key,memory, upload_file,question):
    model = ChatOpenAI(model="deepseek-chat", api_key=api_key,
                       base_url="https://api.deepseek.com")
    file_content=upload_file.read()
    temp_file_path="temp.pdf"
    with open(temp_file_path,"wb") as temp_file:
        temp_file.write(file_content)
    loader=PyPDFLoader(temp_file_path)
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["。","！","？","，","、",""]
    )
    texts=text_splitter.split_documents(docs)
    embeddings_model=DashScopeEmbeddings(model="text-embedding-v1",
                                            dashscope_api_key="sk-e2a1b69874884a6d89d53b47bf86046c")
    db=FAISS.from_documents(texts,embeddings_model)
    retriever = db.as_retriever()
    qa=ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response=qa.invoke({"chat_history":memory,"question":question})
    return response