# --- 9. 節點 (Node)：摘要生成器 (V2.2 最終修復版) ---
def generate_summary_node(state: OmniState) -> Dict[str, Any]:
    print("\n--- [節點 3] AI 正在撰寫摘要 ---")

    # 1. 檢查前一步驟是否有錯誤
    if state.get("error") and not state.get("file_obj"):
        print("偵測到前一步驟錯誤且無備用檔案，跳過生成。")
        return {"summary": f"無法生成摘要，原因：{state['error']}"}

    # 2. 初始化 Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", google_api_key=my_api_key
    )

    # 3. 準備提示詞
    base_requirements = (
        "你是一位全能的資訊整理專家。請為我撰寫一份「懶人包摘要」。"
        "\n\n"
        "【要求】：\n"
        "1. **語言**：無論原文是哪國語言，請全部翻譯並整理成 **繁體中文 (Traditional Chinese)**。\n"
        "2. **格式**：\n"
        "   - **一言以蔽之**：用一句話總結核心主旨。\n"
        "   - **關鍵重點**：列出 3-5 個最重要的資訊點 (Bullet points)。\n"
        "   - **詳細摘要**：針對內容進行邏輯分段的詳細說明。\n"
        "3. **語氣**：專業但輕鬆，適合快速閱讀。"
    )

    try:
        messages = []

        # === 判斷輸入來源 ===
        if state.get("file_obj"):
            # 【情況 A】：有檔案 (音訊)
            print("模式：聽覺處理 (Audio Processing)")

            # --- 增加 Debug 資訊 ---
            file_obj = state["file_obj"]
            print(f"DEBUG: 檔案 URI: {file_obj.uri}")
            print(f"DEBUG: 檔案類型: {file_obj.mime_type}")

            # 確保檔案已經處理完成 (通常音訊很快，但檢查一下比較保險)
            import time

            while file_obj.state.name == "PROCESSING":
                print("DEBUG: Google 正在處理檔案中，等待 2 秒...")
                time.sleep(2)
                file_obj = genai.get_file(file_obj.name)  # 重新整理狀態

            if file_obj.state.name == "FAILED":
                raise ValueError("Google 無法處理此音訊檔案")

            audio_prompt = base_requirements + "\n\n請根據附檔的音訊內容進行摘要。"

            # 【修正重點】：將 "data" 改為 "file_uri"
            message = HumanMessage(
                content=[
                    {"type": "text", "text": audio_prompt},
                    {
                        "type": "media",
                        "mime_type": file_obj.mime_type,
                        "file_uri": file_obj.uri,  # <--- 這裡原本是 "data"，一定要改成 "file_uri"
                    },
                ]
            )
            messages = [message]

        elif state.get("content"):
            # 【情況 B】：有文字 (字幕/網頁)
            print("模式：文字閱讀 (Text Processing)")

            source = state.get("source_type", "unknown")
            text_prompt = (
                base_requirements
                + f"\n\n來源類型：{source}\n【內容】：\n{state['content']}"
            )
            messages = [HumanMessage(content=text_prompt)]

        else:
            return {"summary": "錯誤：沒有內容也沒有檔案可以處理。"}

        # 呼叫 AI
        print("正在呼叫 Gemini 生成摘要 (這可能需要幾秒鐘)...")
        response = llm.invoke(messages)

        print("摘要生成完成！")
        return {"summary": response.content}

    except Exception as e:
        print(f"AI 生成失敗: {e}")
        return {"summary": f"AI 生成失敗: {str(e)}", "error": str(e)}
