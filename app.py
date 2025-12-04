import gradio as gr
import os
import re
from main import app as graph_app


def extract_filename_and_clean_summary(summary_text):
    """
    從摘要中提取 AI 建議的檔名，並回傳 (檔名, 清理後的內容)
    """
    # 預設檔名
    filename = "summary_output"
    cleaned_summary = summary_text

    # 嘗試用 Regex 抓取第一行的 "# 檔名：..."
    # 格式對應 Prompt: # 檔名：[你的檔名]
    match = re.search(r"^# 檔名：(.+)", summary_text.strip())

    if match:
        raw_name = match.group(1).strip()
        # 清理檔名 (只留合法字元)
        filename = re.sub(r"[^\w\u4e00-\u9fa5\-\s]+", "_", raw_name).strip()[:50]

        # (選擇性) 如果你希望文字檔內容不要包含這一行 "檔名：..."，可以在這裡移除
        # 但通常保留當作文件標題也不錯，這裡我選擇保留，但確保它是 H1 標題格式
        cleaned_summary = summary_text  # 不做更動
    else:
        # Fallback: 如果 AI 沒聽話，就抓第一行當檔名，稍微處理一下
        first_line = summary_text.strip().split("\n")[0]
        clean_line = re.sub(r"[^\w\u4e00-\u9fa5]+", "_", first_line)
        filename = clean_line[:30] if clean_line else "summary_output"

    return filename, cleaned_summary


def process_input_stream(user_input, user_api_key):
    """
    使用 generator (yield) 來達成即時串流更新
    """
    if not user_input:
        yield "請輸入網址或文字", "請輸入網址或文字", None
        return

    # 1. 準備輸入
    inputs = {
        "input_text": user_input,
        "api_key": user_api_key if user_api_key.strip() else None,
        "source_type": "",
        "content": None,
        "summary": None,
        "error": None,
        "file_obj": None,
        "logs": [],
    }

    log_content = "🚀 開始執行...\n"
    final_summary = ""

    try:
        # 2. 使用 .stream() 逐步執行
        for event in graph_app.stream(inputs):
            for node_name, updates in event.items():
                if "logs" in updates:
                    new_logs = updates.get("logs", [])
                    # 簡單去重顯示邏輯
                    current_logs_set = set(log_content.strip().split("\n"))
                    for log_line in new_logs:
                        if log_line not in current_logs_set:
                            log_content += f"{log_line}\n"
                            yield log_content, "⏳ 正在思考與撰寫摘要...", None

                if "summary" in updates and updates["summary"]:
                    final_summary = updates["summary"]

        # 3. 執行完成，處理檔案下載
        yield log_content + "\n✅ 執行完畢！", final_summary, None

        # 【V1.2 新邏輯】：使用 AI 建議的檔名
        safe_name, final_content = extract_filename_and_clean_summary(final_summary)
        output_filename = f"{safe_name}.txt"

        # 寫入檔案
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_content)

        yield log_content + f"\n💾 檔案已建立: {output_filename}", final_summary, output_filename

    except Exception as e:
        error_msg = f"發生未知錯誤: {str(e)}"
        yield log_content + f"\n❌ {error_msg}", error_msg, None


# --- 定義 Gradio 介面 ---
with gr.Blocks(title="OmniSummarizer 全能懶人消化器") as demo:
    gr.Markdown("# 🤖 OmniSummarizer V1.2 - 全能懶人消化器")
    gr.Markdown(
        "支援：YouTube (字幕/語音)、網頁文章、純文字 | 自動轉繁體中文 | **AI 智慧取名**"
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                label="輸入來源",
                placeholder="貼上 YouTube 連結、網址，或是一整段文章...",
                lines=5,
            )

            api_key_box = gr.Textbox(
                label="Gemini API Key (選填)",
                placeholder="sk-...",
                type="password",
                info="預設使用 .env 設定。若額度用完，可在此手動輸入新的 Key 覆寫。",
            )

            submit_btn = gr.Button("🚀 開始消化 (Generate)", variant="primary")

            log_box = gr.Textbox(
                label="執行日誌 (Process Logs)",
                value="準備就緒...",
                lines=10,
                max_lines=15,
                interactive=False,
            )

        with gr.Column(scale=3):
            output_text = gr.Markdown(label="懶人包摘要")
            download_file = gr.File(label="下載摘要 (.txt)")

    submit_btn.click(
        fn=process_input_stream,
        inputs=[input_box, api_key_box],
        outputs=[log_box, output_text, download_file],
    )

if __name__ == "__main__":
    demo.launch()
