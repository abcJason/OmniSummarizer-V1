import os
import re
import time
from dotenv import load_dotenv

import google.generativeai as genai
import yt_dlp

# --- 1. 初始化環境 ---
load_dotenv()
default_api_key = os.environ.get("MY_GEMINI_KEY")

from typing import Optional, List, Dict, Any, Annotated
from typing_extensions import TypedDict
import operator  # 用來做列表的合併 (append)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# --- 3. 定義狀態 (State) ---
class OmniState(TypedDict):
    input_text: str
    source_type: str
    content: Optional[str]
    summary: Optional[str]
    error: Optional[str]
    file_obj: Any
    api_key: Optional[str]
    # 【新增】日誌列表：使用 operator.add 讓每個節點的回傳值自動 "Append" 進去，而不是覆蓋
    logs: Annotated[List[str], operator.add]


# --- 4. 輔助函式 ---
def detect_source_type(input_text: str) -> str:
    text = input_text.strip().lower()
    if "youtube.com" in text or "youtu.be" in text:
        return "youtube"
    elif text.startswith("http://") or text.startswith("https://"):
        return "web"
    else:
        return "text"


# --- 4.5 新增輔助函式：提取 Video ID ---
def extract_video_id(url: str) -> Optional[str]:
    """從 YouTube 網址中提取 Video ID"""
    # 支援 https://www.youtube.com/watch?v=VIDEO_ID 和 https://youtu.be/VIDEO_ID
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# --- 5. 節點：路由器 ---
def analyze_input_node(state: OmniState) -> Dict[str, Any]:
    source_type = detect_source_type(state["input_text"])
    log_msg = f"--- [節點 1] 分析輸入 ---\n偵測結果: {source_type}"
    print(log_msg)  # 保留後台 print 方便除錯
    return {"source_type": source_type, "logs": [log_msg]}


# --- 6. 節點：YouTube (V2.1 yt-dlp 全能版) ---
def load_youtube_node(state: OmniState) -> Dict[str, Any]:
    url = state["input_text"]
    logs = ["--- [節點 2-A] 處理 YouTube ---"]

    # 設定 API Key
    current_key = state.get("api_key") or default_api_key
    if current_key:
        genai.configure(api_key=current_key)

    # === Plan A: 使用 yt-dlp 下載字幕 (省 Token 模式) ===
    logs.append("嘗試使用 yt-dlp 下載字幕 (Plan A)...")

    # 產生唯一的暫存檔前綴 (避免多執行緒衝突)
    import uuid

    file_prefix = f"sub_{uuid.uuid4().hex[:8]}"

    # yt-dlp 設定：只抓字幕，不抓影片
    ydl_opts_sub = {
        "skip_download": True,  # 關鍵：不下載影片檔
        "writeautomaticsub": True,  # 嘗試抓自動產生的字幕 (通常都有)
        "writesubtitles": True,  # 嘗試抓手動上傳的字幕
        "sublangs": [
            "zh-Hant",
            "zh-TW",
            "zh",
            "en",
            "en-US",
        ],  # 優先抓繁中，沒有就抓英文
        "outtmpl": file_prefix,  # 輸出檔名模板
        "quiet": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts_sub) as ydl:
            ydl.download([url])

        # 檢查下載了什麼檔案 (.vtt)
        # yt-dlp 會自動加上語言後綴，例如 sub_xxx.zh-Hant.vtt 或 sub_xxx.en.vtt
        generated_files = [
            f
            for f in os.listdir(".")
            if f.startswith(file_prefix) and f.endswith(".vtt")
        ]

        if generated_files:
            # 找到字幕檔了！
            sub_file = generated_files[0]  # 抓第一個找到的
            logs.append(f"✅ 成功下載字幕檔: {sub_file}")

            # 讀取並清洗 VTT 格式 (去除時間軸，只留文字)
            clean_text = []
            with open(sub_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 過濾掉 WEBVTT 標頭、空行、時間軸 (例如 00:00:01.000 --> ...)
                    if "-->" in line or line == "WEBVTT" or not line:
                        continue
                    # 過濾掉重複的行 (有些字幕會重複上一句)
                    if clean_text and clean_text[-1] == line:
                        continue
                    # 移除一些 HTML 標籤 (如 <c.colorE5E5E5>)
                    line = re.sub(r"<[^>]+>", "", line)
                    if line:
                        clean_text.append(line)

            # 刪除暫存的 .vtt 檔
            for f in generated_files:
                os.remove(f)

            full_transcript = "\n".join(clean_text)

            # 檢查字數，如果太少可能是空的
            if len(full_transcript) > 50:
                msg = f"✅ 字幕清洗完成，長度: {len(full_transcript)} 字"
                logs.append(msg)
                return {
                    "content": f"【影片字幕】：\n{full_transcript}",
                    "file_obj": None,
                    "error": None,
                    "logs": logs,
                }
            else:
                logs.append("⚠️ 下載的字幕內容過短，視為失敗。")
        else:
            logs.append("⚠️ yt-dlp 執行完畢但未發現字幕檔 (可能該影片無字幕)。")

    except Exception as e:
        logs.append(f"⚠️ Plan A 字幕下載失敗: {e}")
        # 清理可能殘留的檔案
        for f in os.listdir("."):
            if f.startswith(file_prefix):
                try:
                    os.remove(f)
                except:
                    pass

    # === Plan B: 音訊 (Gemini 聽力模式) ===
    if not current_key:
        err = "無字幕且未提供 API Key，無法進行音訊處理。"
        logs.append(f"❌ {err}")
        return {"error": err, "content": None, "logs": logs}

    logs.append("⚠️ 啟動 Plan B：下載音訊 (Gemini 聽力模式)...")
    temp_audio_file = f"audio_{uuid.uuid4().hex[:8]}.m4a"
    ydl_opts_audio = {
        "format": "bestaudio[ext=m4a]/best",
        "outtmpl": temp_audio_file,
        "quiet": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
            ydl.download([url])
        logs.append("音訊下載完成，正在上傳到 Gemini...")
        audio_file = genai.upload_file(path=temp_audio_file)

        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

        logs.append(f"✅ 上傳成功 (URI: {audio_file.uri})")
        return {"content": None, "file_obj": audio_file, "error": None, "logs": logs}
    except Exception as e:
        logs.append(f"❌ YouTube 音訊處理也失敗: {str(e)}")
        return {"error": str(e), "content": None, "file_obj": None, "logs": logs}


# --- 7. 節點：網頁 ---
def load_web_node(state: OmniState) -> Dict[str, Any]:
    logs = ["--- [節點 2-B] 處理網頁 ---"]
    try:
        loader = WebBaseLoader(state["input_text"])
        docs = loader.load()
        if not docs:
            return {
                "error": "網頁抓取為空",
                "content": None,
                "logs": logs + ["❌ 網頁抓取為空"],
            }
        clean_content = re.sub(r"\n\s*\n", "\n\n", docs[0].page_content)
        logs.append(f"✅ 成功抓取網頁，長度: {len(clean_content)} 字")
        return {"content": clean_content, "error": None, "logs": logs}
    except Exception as e:
        return {
            "error": str(e),
            "content": None,
            "logs": logs + [f"❌ 網頁處理失敗: {str(e)}"],
        }


# --- 8. 節點：純文字 ---
def load_text_node(state: OmniState) -> Dict[str, Any]:
    return {
        "content": state["input_text"],
        "error": None,
        "logs": ["--- [節點 2-C] 處理純文字 ---"],
    }


# --- 9. 節點：摘要生成 ---
def generate_summary_node(state: OmniState) -> Dict[str, Any]:
    logs = ["\n--- [節點 3] AI 正在撰寫摘要 ---"]

    if state.get("error") and not state.get("file_obj"):
        return {
            "summary": f"無法生成: {state['error']}",
            "logs": logs + ["偵測到前一步驟錯誤，跳過。"],
        }

    # 【API Key 檢查與回報】
    current_key = state.get("api_key") or default_api_key
    if not current_key:
        return {
            "summary": "錯誤：未設定 API Key",
            "error": "No API Key",
            "logs": logs + ["❌ 錯誤：找不到 API Key"],
        }

    # 顯示目前使用的 Key 來源 (遮罩處理)
    key_source = "手動輸入" if state.get("api_key") else "預設 .env"
    masked_key = current_key[:4] + "*" * 10 + current_key[-4:]
    logs.append(f"🔑 使用 Key: {masked_key} ({key_source})")

    genai.configure(api_key=current_key)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", google_api_key=current_key
    )

    base_requirements = (
        "你是一位全能的資訊整理專家。請為我撰寫一份「懶人包摘要」。"
        "\n\n【要求】：\n"
        "1. **檔名指令(重要)**：請根據內容，取一個最適合存檔的檔名。並在回應的**第一行**，嚴格依照此格式輸出：`# 檔名：[你的檔名]`。\n"
        "2. **直接輸出**：請直接開始輸出內容，**絕對不要**有任何開場白（如「好的」、「這是我整理的...」等廢話）。\n"
        "3. **語言**：全部翻譯並整理成 **繁體中文**。\n"
        "4. **格式**：\n"
        "   - **一言以蔽之**：用一句話總結核心主旨。\n"
        "   - **關鍵重點**：列出 3-5 個最重要的資訊點 (Bullet points)。\n"
        "   - **詳細摘要**：針對內容進行邏輯分段的詳細說明。\n"
        "5. **語氣**：專業但輕鬆。"
    )

    try:
        messages = []
        if state.get("file_obj"):
            logs.append("模式：聽覺處理 (Audio)")
            file_obj = state["file_obj"]
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
            logs.append("模式：文字閱讀 (Text)")
            text_prompt = (
                base_requirements
                + f"\n\n來源：{state.get('source_type')}\n【內容】：\n{state['content']}"
            )
            messages = [HumanMessage(content=text_prompt)]
        else:
            return {"summary": "無內容可處理", "logs": logs}

        logs.append("🚀 正在呼叫 Gemini 生成摘要...")
        response = llm.invoke(messages)
        logs.append("✅ 摘要生成完成！")
        return {"summary": response.content, "logs": logs}
    except Exception as e:
        return {
            "summary": f"AI 生成失敗: {str(e)}",
            "error": str(e),
            "logs": logs + [f"❌ AI 生成失敗: {str(e)}"],
        }


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
