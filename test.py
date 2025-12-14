import os
import re
import time
from dotenv import load_dotenv

# Google 原生 SDK (用來上傳檔案)
import google.generativeai as genai
import yt_dlp

# --- 1. 初始化環境 ---
load_dotenv()
default_api_key = os.environ.get("MY_GEMINI_KEY")
# 注意：這裡我們先不報錯退出，因為使用者可能想在 UI 手動輸入 Key

# --- 2. 匯入必要的 LangChain 與工具庫 ---
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 網頁與 YouTube 載入器
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeLoader

from langchain_core.messages import HumanMessage

# LangGraph 元件
from langgraph.graph import StateGraph, END


# --- 3. 定義狀態 (State) ---
# 這是我們 V1 版本的資料結構，所有節點都只會讀寫這個字典
class OmniState(TypedDict):
    input_text: str
    source_type: str
    content: Optional[str]
    summary: Optional[str]
    error: Optional[str]
    file_obj: Any
    api_key: Optional[str]  # 【新增】動態 API Key


# --- 4. 輔助函式 ---
def detect_source_type(input_text: str) -> str:
    """
    簡單的規則判斷：
    1. 包含 youtube.com 或 youtu.be -> 'youtube'
    2. 包含 http:// 或 https:// -> 'web'
    3. 其他 -> 'text'
    """
    text = input_text.strip().lower()
    if "youtube.com" in text or "youtu.be" in text:
        return "youtube"
    elif text.startswith("http://") or text.startswith("https://"):
        return "web"
    else:
        return "text"


# --- 5. 節點：路由器 ---
def analyze_input_node(state: OmniState) -> Dict[str, Any]:
    print("\n--- [節點 1] 分析輸入類型 ---")
    source_type = detect_source_type(state["input_text"])
    print(f"偵測結果: {source_type}")
    return {"source_type": source_type}


# --- 6. 節點：YouTube ---
def load_youtube_node(state: OmniState) -> Dict[str, Any]:
    print("\n--- [節點 2-A] 處理 YouTube ---")
    url = state["input_text"]

    # 確保使用正確的 Key
    current_key = state.get("api_key") or default_api_key
    if current_key:
        genai.configure(api_key=current_key)

    # Plan A: 字幕
    try:
        print("1. 嘗試下載字幕...")
        common_languages = [
            "zh-Hant",
            "zh-TW",
            "zh-Hans",
            "zh",
            "zh-HK",
            "en",
            "en-US",
            "en-GB",
            "ja",
            "ko",
            "es",
            "fr",
            "de",
        ]
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=False, language=common_languages
        )
        docs = loader.load()
        if docs:
            transcript = docs[0].page_content
            print(f"✅ 成功抓取字幕，長度: {len(transcript)} 字")
            return {
                "content": f"【影片字幕】：\n{transcript}",
                "file_obj": None,
                "error": None,
            }
    except Exception as e:
        print(f"⚠️ 字幕抓取失敗 (轉 Plan B): {e}")

    # Plan B: 音訊
    if not current_key:
        return {"error": "無字幕且未提供 API Key，無法進行音訊處理。", "content": None}

    print("2. 啟動 Plan B：下載音訊...")
    temp_audio_file = f"temp_{int(time.time())}.m4a"  # 避免檔名衝突
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/best",
        "outtmpl": temp_audio_file,
        "quiet": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("3. 上傳音訊到 Gemini...")
        audio_file = genai.upload_file(path=temp_audio_file)

        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

        return {"content": None, "file_obj": audio_file, "error": None}
    except Exception as e:
        return {
            "error": f"YouTube 處理失敗: {str(e)}",
            "content": None,
            "file_obj": None,
        }


# --- 7. 節點：網頁 ---
def load_web_node(state: OmniState) -> Dict[str, Any]:
    print("\n--- [節點 2-B] 處理網頁 ---")
    try:
        loader = WebBaseLoader(state["input_text"])
        docs = loader.load()
        if not docs:
            return {"error": "網頁抓取為空", "content": None}
        clean_content = re.sub(r"\n\s*\n", "\n\n", docs[0].page_content)
        return {"content": clean_content, "error": None}
    except Exception as e:
        return {"error": f"網頁處理失敗: {str(e)}", "content": None}


# --- 8. 節點：純文字 ---
def load_text_node(state: OmniState) -> Dict[str, Any]:
    return {"content": state["input_text"], "error": None}


# --- 9. 節點：摘要生成 ---
def generate_summary_node(state: OmniState) -> Dict[str, Any]:
    print("\n--- [節點 3] AI 正在撰寫摘要 ---")
    if state.get("error") and not state.get("file_obj"):
        return {"summary": f"無法生成: {state['error']}"}

    # 【關鍵】：優先使用 User 輸入的 Key
    current_key = state.get("api_key") or default_api_key
    if not current_key:
        return {"summary": "錯誤：未設定 Gemini API Key", "error": "No API Key"}

    genai.configure(api_key=current_key)  # 確保 genai sdk 也更新
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", google_api_key=current_key
    )

    base_requirements = (
        "你是一位全能的資訊整理專家。請為我撰寫一份「懶人包摘要」。"
        "\n\n【要求】：\n"
        "1. **語言**：全部翻譯並整理成 **繁體中文**。\n"
        "2. **格式**：一言以蔽之、關鍵重點(Bullet points)、詳細摘要。\n"
        "3. **語氣**：專業但輕鬆。"
    )

    try:
        messages = []
        if state.get("file_obj"):
            print("模式：聽覺處理")
            file_obj = state["file_obj"]
            # 等待處理完成
            while file_obj.state.name == "PROCESSING":
                time.sleep(1)
                file_obj = genai.get_file(file_obj.name)

            if file_obj.state.name == "FAILED":
                raise ValueError("Google 處理檔案失敗")

            audio_prompt = base_requirements + "\n\n請根據附檔音訊摘要。"
            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": audio_prompt},
                        {
                            "type": "media",
                            "mime_type": file_obj.mime_type,
                            "file_uri": file_obj.uri,
                        },
                    ]
                )
            ]
        elif state.get("content"):
            print("模式：文字閱讀")
            text_prompt = (
                base_requirements
                + f"\n\n來源：{state.get('source_type')}\n【內容】：\n{state['content']}"
            )
            messages = [HumanMessage(content=text_prompt)]
        else:
            return {"summary": "無內容可處理"}

        response = llm.invoke(messages)
        return {"summary": response.content}
    except Exception as e:
        return {"summary": f"AI 生成失敗: {str(e)}", "error": str(e)}


# --- 10. 路由與組裝 ---
def route_based_on_source(state: OmniState) -> str:
    source = state["source_type"]
    if source == "youtube":
        return "process_youtube"
    elif source == "web":
        return "process_web"
    else:
        return "process_text"


workflow = StateGraph(OmniState)
workflow.add_node("analyze_node", analyze_input_node)
workflow.add_node("youtube_node", load_youtube_node)
workflow.add_node("web_node", load_web_node)
workflow.add_node("text_node", load_text_node)
workflow.add_node("summarize_node", generate_summary_node)

workflow.set_entry_point("analyze_node")
workflow.add_conditional_edges(
    "analyze_node",
    route_based_on_source,
    {
        "process_youtube": "youtube_node",
        "process_web": "web_node",
        "process_text": "text_node",
    },
)
workflow.add_edge("youtube_node", "summarize_node")
workflow.add_edge("web_node", "summarize_node")
workflow.add_edge("text_node", "summarize_node")
workflow.add_edge("summarize_node", END)

app = workflow.compile()
