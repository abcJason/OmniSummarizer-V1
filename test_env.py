import os
import sys
from dotenv import load_dotenv

# 1. 測試環境變數
load_dotenv()
my_key = os.environ.get("MY_GEMINI_KEY")

print(f"Python 版本: {sys.version}")
print(f"API Key 讀取: {'成功' if my_key else '失敗'}")

try:
    # 2. 測試 LangChain 與 Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=my_key)
    print("Gemini 模型初始化: 成功")

    # 3. 測試 LangGraph 匯入
    from langgraph.graph import StateGraph

    print("LangGraph 匯入: 成功")

    # 4. 測試 YouTube 工具匯入
    from youtube_transcript_api import YouTubeTranscriptApi
    from langchain_community.document_loaders import YoutubeLoader

    print("YouTube 工具匯入: 成功")

    # 5. 測試 Web 工具匯入
    from bs4 import BeautifulSoup

    print("Web 工具匯入: 成功")

    print("\n✅ 環境建置完美！可以開始開發了！")

except Exception as e:
    print(f"\n❌ 環境有問題: {e}")
