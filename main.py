import os
import re  # 用來做正規表達式判斷 (Regex)
from dotenv import load_dotenv

# 新增這個：Google 原生 SDK (用來上傳檔案)
import google.generativeai as genai
import yt_dlp  # 用來下載 YouTube 音軌

# --- 1. 初始化環境與 API Key ---
load_dotenv()
my_api_key = os.environ.get("MY_GEMINI_KEY")
if not my_api_key:
    print("❌ 錯誤：找不到 MY_GEMINI_KEY")
    exit(1)

# 【新增】：設定 Google GenAI SDK
genai.configure(api_key=my_api_key)

# --- 2. 匯入必要的 LangChain 與工具庫 ---
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 網頁與 YouTube 載入器
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeLoader

from langchain_core.messages import HumanMessage  # 新增這個：用來建構多模態訊息

# LangGraph 元件
from langgraph.graph import StateGraph, END


# --- 3. 定義狀態 (State) ---
# 這是我們 V1 版本的資料結構，所有節點都只會讀寫這個字典
class OmniState(TypedDict):
    input_text: str  # 使用者最原始的輸入 (網址或文字)
    source_type: str  # 判斷結果：'youtube', 'web', 'text'
    content: Optional[str]  # 抓取並清洗後的「純文字」內容
    summary: Optional[str]  # 最終生成的摘要
    error: Optional[str]  # 如果中間出錯，記錄錯誤訊息
    file_obj: Any  # 存放上傳後的檔案物件 (如果是音訊處理的話)


# --- 4. 輔助函式：判斷輸入類型 ---
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


# --- 5. 節點 (Node)：路由器 (分析輸入) ---
def analyze_input_node(state: OmniState) -> Dict[str, Any]:
    print("\n--- [節點 1] 分析輸入類型 ---")
    user_input = state["input_text"]
    source_type = detect_source_type(user_input)

    print(f"偵測結果: {source_type}")
    # 回傳更新 state
    return {"source_type": source_type}


# --- 6. 節點 (Node)：YouTube 載入器 ---
def load_youtube_node(state: OmniState) -> Dict[str, Any]:
    print("\n--- [節點 2-A] 處理 YouTube ---")
    url = state["input_text"]

    try:
        print("1. 嘗試下載字幕...")
        # 優先找中文，然後英文，接著是日韓法德西俄等常見語言
        common_languages = [
            "zh-Hant",
            "zh-TW",
            "zh-Hans",
            "zh",
            "zh-HK",
            "en",
            "ja",
            "ko",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
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
        print(f"⚠️ 字幕抓取失敗 (將嘗試 Plan B): {e}")

    # === Plan B: 下載音訊並「聽」內容 ===
    print("2. 啟動 Plan B：下載音訊 (Gemini 聽力模式)...")

    # 設定下載檔名 (暫存)
    temp_audio_file = "temp_audio.m4a"

    # yt-dlp 設定：只下載最好的音訊，並存成 m4a
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": temp_audio_file,
        "quiet": True,
        "noplaylist": True,
    }

    try:
        # 1. 下載音訊
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✅ 音訊下載完成: {temp_audio_file}")

        # 2. 上傳到 Gemini
        print("3. 正在上傳音訊到 Google Gemini...")
        audio_file = genai.upload_file(path=temp_audio_file)
        print(f"✅ 上傳成功，File URI: {audio_file.uri}")

        # 3. 刪除本地暫存檔 (保持環境乾淨)
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

        # 回傳 file_obj 讓下一個節點使用
        return {"content": None, "file_obj": audio_file, "error": None}

    except Exception as e:
        print(f"❌ Plan B 失敗: {e}")
        return {
            "error": f"YouTube 字幕與音訊皆失敗: {str(e)}",
            "content": None,
            "file_obj": None,
        }


# --- 7. 節點 (Node)：網頁載入器 ---
def load_web_node(state: OmniState) -> Dict[str, Any]:
    print("\n--- [節點 2-B] 處理網頁 ---")
    url = state["input_text"]

    try:
        # 使用 WebBaseLoader 抓取網頁
        loader = WebBaseLoader(url)
        docs = loader.load()

        if not docs:
            return {"error": "網頁抓取為空", "content": None}

        # 簡單清洗：去除多餘換行
        raw_content = docs[0].page_content
        clean_content = re.sub(r"\n\s*\n", "\n\n", raw_content)  # 把多個空行變成一個

        print(f"成功抓取網頁，長度: {len(clean_content)} 字")
        return {"content": clean_content, "error": None}

    except Exception as e:
        print(f"網頁載入失敗: {e}")
        return {"error": f"網頁處理失敗: {str(e)}", "content": None}


# --- 8. 節點 (Node)：純文字處理 (透傳) ---
def load_text_node(state: OmniState) -> Dict[str, Any]:
    print("\n--- [節點 2-C] 處理純文字 ---")
    # 如果使用者直接貼文章，就直接當作 content
    return {"content": state["input_text"], "error": None}


# --- 9. 節點 (Node)：摘要生成器 (AI 大腦) ---
def generate_summary_node(state: OmniState) -> Dict[str, Any]:
    print("\n--- [節點 3] AI 正在撰寫摘要 ---")

    # 1. 檢查前一步驟是否有錯誤
    if state.get("error"):
        print("偵測到前一步驟錯誤，跳過生成。")
        return {"summary": f"無法生成摘要，原因：{state['error']}"}

    # 2. 初始化 Gemini
    # 注意：這裡一定要傳入 google_api_key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", google_api_key=my_api_key
    )

    # 3. 設定提示詞 (Prompt)
    # 我們明確要求：不管原文是什麼，都要用繁體中文回答
    system_prompt = (
        "你是一位全能的資訊整理專家。請閱讀以下內容（來源：{source_type}），"
        "並為我撰寫一份「懶人包摘要」。"
        "\n\n"
        "【要求】：\n"
        "1. **語言**：無論原文是哪國語言，請全部翻譯並整理成 **繁體中文 (Traditional Chinese)**。\n"
        "2. **格式**：\n"
        "   - **一言以蔽之**：用一句話總結核心主旨。\n"
        "   - **關鍵重點**：列出 3-5 個最重要的資訊點 (Bullet points)。\n"
        "   - **詳細摘要**：針對內容進行邏輯分段的詳細說明。\n"
        "3. **語氣**：專業但輕鬆，適合快速閱讀。"
        "\n\n"
        "【內容】：\n{content}"
    )

    try:
        messages = []

        # === 判斷輸入來源 ===
        if state.get("file_obj"):
            # 【情況 A】：有檔案 (音訊)
            print("模式：聽覺處理 (Audio Processing)")
            # 這是 LangChain 傳遞多模態檔案的標準寫法
            message = HumanMessage(
                content=[
                    {"type": "text", "text": system_prompt},
                    {
                        "type": "media",
                        "mime_type": state["file_obj"].mime_type,
                        "data": state["file_obj"].uri,
                    },
                ]
            )
            messages = [message]

        elif state.get("content"):
            # 【情況 B】：有文字 (字幕/網頁)
            print("模式：文字閱讀 (Text Processing)")
            messages = [
                HumanMessage(
                    content=system_prompt + f"\n\n【內容】：\n{state['content']}"
                )
            ]
        else:
            return {"summary": "錯誤：沒有內容也沒有檔案可以處理。"}

        # 呼叫 AI
        response = llm.invoke(messages)

        print("摘要生成完成！")
        return {"summary": response.content}

    except Exception as e:
        print(f"AI 生成失敗: {e}")
        return {"summary": f"AI 生成失敗: {str(e)}", "error": str(e)}


# --- 10. 條件邊邏輯 (Router Logic) ---
def route_based_on_source(state: OmniState) -> str:
    """決定分析完輸入後，要走哪條路"""
    source = state["source_type"]
    if source == "youtube":
        return "process_youtube"
    elif source == "web":
        return "process_web"
    else:
        return "process_text"


# --- 11. 組裝 LangGraph ---

workflow = StateGraph(OmniState)

# (1) 加入所有節點
workflow.add_node("analyze_node", analyze_input_node)
workflow.add_node("youtube_node", load_youtube_node)
workflow.add_node("web_node", load_web_node)
workflow.add_node("text_node", load_text_node)
workflow.add_node("summarize_node", generate_summary_node)

# (2) 設定起點
workflow.set_entry_point("analyze_node")

# (3) 設定條件邊 (從分析節點出發，分三路)
workflow.add_conditional_edges(
    "analyze_node",
    route_based_on_source,
    {
        "process_youtube": "youtube_node",
        "process_web": "web_node",
        "process_text": "text_node",
    },
)

# (4) 設定匯聚邊 (三條路最後都匯聚到 摘要節點)
workflow.add_edge("youtube_node", "summarize_node")
workflow.add_edge("web_node", "summarize_node")
workflow.add_edge("text_node", "summarize_node")

# (5) 設定終點
workflow.add_edge("summarize_node", END)

# (6) 編譯應用程式
app = workflow.compile()


# --- 12. 最終執行測試 ---
if __name__ == "__main__":
    print("\n🚀 OmniSummarizer V1 啟動！")

    # --- 測試案例 ---
    # 案例 A: 你的 YouTube 影片 (測試多語言翻譯能力 + 字幕抓取)
    input_data = "https://www.youtube.com/watch?v=M89pzPpyzpg"

    # 案例 B: 網頁 (你可以把上面註解掉，換測這個)
    # input_data = "https://blog.langchain.dev/langgraph-multi-agent-workflows/"

    print(f"正在處理: {input_data}")

    inputs = {
        "input_text": input_data,
        "source_type": "",
        "content": None,
        "summary": None,
        "error": None,
        "file_obj": None,
    }

    try:
        # 執行 LangGraph
        result = app.invoke(inputs)

        print("\n\n" + "=" * 30)
        print("🌟 最終懶人包產出 🌟")
        print("=" * 30 + "\n")

        if result["error"]:
            print(f"❌ 發生錯誤: {result['error']}")
        else:
            print(result["summary"])

    except Exception as e:
        print(f"程式執行錯誤: {e}")
