import os
import base64
import csv
from io import StringIO

import pandas as pd
import streamlit as st
from openai import OpenAI


# =========================
# 基本設定
# =========================
st.set_page_config(page_title="Anatomy Card Maker", layout="wide")

st.title("Anatomy Card Maker")
st.write("上傳解剖圖 → AI 擷取中英配對 → 網路校正 → 下載 Anki CSV")


# =========================
# API Key
# =========================
api_key = os.environ.get("OPENAI_API_KEY", "").strip()

if not api_key.startswith("sk-"):
    st.error("OPENAI_API_KEY 沒設定好。請先在 terminal 設定 API key。")
    st.code('$env:OPENAI_API_KEY="你的 API key"', language="powershell")
    st.stop()

client = OpenAI(api_key=api_key)


# =========================
# 工具函式
# =========================
def image_to_base64(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


def parse_csv_text(csv_text):
    """
    期待 AI 回傳：
    English,AI_Chinese,Checked_Chinese,Status,Note
    """
    rows = []

    reader = csv.reader(StringIO(csv_text.strip()))
    for row in reader:
        if not row:
            continue

        # 跳過 header
        if row[0].strip().lower() == "english":
            continue

        # 補齊欄位
        while len(row) < 5:
            row.append("")

        english = row[0].strip()
        ai_chinese = row[1].strip()
        checked_chinese = row[2].strip()
        status = row[3].strip()
        note = row[4].strip()

        if english:
            rows.append({
                "English": english,
                "AI_Chinese": ai_chinese,
                "Checked_Chinese": checked_chinese,
                "Status": status,
                "Note": note
            })

    return pd.DataFrame(rows)


def build_anki_cards(df, use_checked=True, card_mode="雙向混合"):
    cards = []

    for _, row in df.iterrows():
        eng = str(row["English"]).strip()
        ai_chi = str(row["AI_Chinese"]).strip()
        checked = str(row["Checked_Chinese"]).strip()

        chi = checked if use_checked and checked else ai_chi

        if not eng or not chi:
            continue

        if card_mode == "英翻中":
            cards.append([eng, chi])

        elif card_mode == "中翻英":
            cards.append([chi, eng])

        elif card_mode == "雙向混合":
            cards.append([eng, chi])
            cards.append([chi, eng])

    return pd.DataFrame(cards, columns=["Front", "Back"])


# =========================
# UI
# =========================
uploaded_file = st.file_uploader(
    "上傳解剖圖片",
    type=["jpg", "jpeg", "png"]
)

col_left, col_right = st.columns([1, 1])

if uploaded_file is not None:
    with col_left:
        st.subheader("圖片預覽")
        st.image(uploaded_file, use_container_width=True)

    with col_right:
        st.subheader("設定")

        card_mode = st.radio(
        "練習模式",
        ["英翻中", "中翻英", "雙向混合"],
        index=2
    )

        random_mode = st.checkbox("隨機出題", value=True)

        use_checked = st.checkbox("下載時使用校正後中文", value=True)

        model_name = st.selectbox(
            "模型",
            ["gpt-5.2", "gpt-5.5"],
            index=0
        )

        run_button = st.button("AI 擷取 + 網路校正", type="primary")


    if run_button and "anki_df" not in st.session_state:
        image_base64 = image_to_base64(uploaded_file)

        prompt = """
你是一個解剖學名詞整理與校對助手。

請完成兩件事：

第一步：
從圖片中擷取所有可見的英文解剖構造名稱與中文翻譯。

第二步：
使用網路資料交叉檢查每一組中英配對是否正確。
如果配對正確，Status 設為 OK。
如果中文翻譯明顯錯誤，Status 設為 FIX，Checked_Chinese 填入較標準的中文翻譯。
如果無法確定，Status 設為 UNKNOWN。

重要規則：
1. 只保留解剖構造名稱，例如 muscle, bone, tendon, ligament, vein, nerve 等。
2. 不要保留區域標題，例如 Head, Neck, Shoulder, Arm, Forearm, Leg, Superficial Muscles, Deeper Muscles。
3. 不要保留頁碼、章節標題、表格標題、圖片說明句。
4. 若圖片中的中文清楚可見，AI_Chinese 請填圖片原本中文。
5. 若圖片中的中文看不清楚，可以留空，但仍可用網路查核 Checked_Chinese。
6. 請只輸出 CSV，不要加任何解釋文字。

CSV 欄位固定為：
English,AI_Chinese,Checked_Chinese,Status,Note

Status 只能使用：
OK
FIX
UNKNOWN

Note 簡短說明即可。
"""

        with st.spinner("AI 正在擷取與查核，可能需要一點時間..."):
            response = client.responses.create(
                model=model_name,
                tools=[
                    {"type": "web_search"}
                ],
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        ],
                    }
                ],
            )

        result = response.output_text.strip()
        st.session_state["ai_done"] = True
        st.session_state["result"] = result
        st.subheader("AI 原始輸出")
        st.text_area("CSV 原文", result, height=250)

        df = parse_csv_text(result)
        st.session_state["df"] = df

        if df.empty:
            st.error("AI 沒有回傳可解析的 CSV，請換張清楚一點的圖片或再試一次。")
            st.stop()

        st.subheader("全部擷取結果")
        st.dataframe(df, use_container_width=True)

        # 只顯示有疑慮的
        suspicious_df = df[df["Status"].str.upper().isin(["FIX", "UNKNOWN"])]
        st.session_state["suspicious_df"] = suspicious_df

        st.subheader("有疑慮的翻譯")
        if suspicious_df.empty:
            st.success("目前沒有發現明顯有疑慮的翻譯。")
        else:
            st.warning(f"發現 {len(suspicious_df)} 筆需要確認")
            st.dataframe(suspicious_df, use_container_width=True)



        # 建立 Anki CSV
        anki_df = build_anki_cards(
            df,
            use_checked=use_checked,
            card_mode=card_mode
        )

        if "anki_df" not in st.session_state:
            anki_df = build_anki_cards(
                df,
                use_checked=use_checked,
                card_mode=card_mode
            )

            if random_mode:
                anki_df = anki_df.sample(frac=1).reset_index(drop=True)

            st.session_state["anki_df"] = anki_df

        
        st.session_state["anki_df"] = anki_df
        st.session_state["cards"] = anki_df.values.tolist()
        st.session_state["card_index"] = 0
        st.session_state["show_answer"] = False
        st.session_state["levels"] = {}

        st.subheader("Anki 卡片預覽")
        st.dataframe(anki_df, use_container_width=True)

        csv_data = anki_df.to_csv(
            index=False,
            header=False
        ).encode("utf-8-sig")

        st.download_button(
            label="下載 Anki CSV",
            data=csv_data,
            file_name="anki_anatomy_cards.csv",
            mime="text/csv"
        )

# =========================
# 線上單字卡
# =========================

st.subheader("線上單字卡練習")

if "anki_df" not in st.session_state:
    st.info("先上傳圖片並按『AI 擷取 + 網路校正』，下面才會出現單字卡")

else:
    # 建立 cards
    if "cards" not in st.session_state:
        st.session_state["cards"] = st.session_state["anki_df"].values.tolist()

    cards = st.session_state["cards"]

    if not cards:
        st.warning("目前沒有可用卡片")
        st.stop()

    # 初始化狀態
    if "show_answer" not in st.session_state:
        st.session_state.show_answer = False

    if "levels" not in st.session_state:
        st.session_state.levels = {}

    if "schedule" not in st.session_state:
        st.session_state.schedule = {i: 0 for i in range(len(cards))}

    if "card_status" not in st.session_state:
        st.session_state.card_status = {i: "new" for i in range(len(cards))}

    if "review_count" not in st.session_state:
        st.session_state.review_count = 0

    # =========================
    # 統計顏色數量
    # =========================
    status_counts = {"new": 0, "learning": 0, "review": 0, "done": 0}

    for s in st.session_state.card_status.values():
        if s in status_counts:
            status_counts[s] += 1

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"### 🔵 新卡：{status_counts['new']}")
    with c2:
        st.markdown(f"### 🔴 學習中：{status_counts['learning']}")
    with c3:
        st.markdown(f"### 🟢 複習：{status_counts['review']}")

    st.write(f"已練習：{st.session_state.review_count} 次")

    # =========================
    # 完成判斷
    # =========================
    if (
        status_counts["new"] == 0
        and status_counts["learning"] == 0
        and status_counts["review"] == 0
    ):
        st.success("今天的進度完成！")
        st.stop()

    # =========================
    # 找下一張卡
    # =========================
    def get_next_card():
        schedule = st.session_state.schedule
        card_status = st.session_state.card_status

        available = [i for i in range(len(cards)) if card_status[i] != "done"]

        if not available:
            return None

        return min(available, key=lambda i: schedule[i])

    current_index = get_next_card()

    if current_index is None:
        st.success("今天的進度完成！")
        st.stop()

    st.session_state["current_index"] = current_index

    front, back = cards[current_index]

    current_status = st.session_state.card_status[current_index]

    # 顯示顏色狀態
    if current_status == "new":
        st.markdown("### 🔵 新卡")
    elif current_status == "learning":
        st.markdown("### 🔴 學習中")
    elif current_status == "review":
        st.markdown("### 🟢 複習")

    # 顯示卡片
    if st.session_state.show_answer:
        card_text = back
        label = "答案"
    else:
        card_text = front
        label = "題目"

    st.markdown(f"#### {label}")
    st.markdown(f"## {card_text}")

    st.write(f"第 {current_index + 1} / {len(cards)} 張")

    # 翻面
    if st.button("翻面 / 顯示答案"):
        st.session_state.show_answer = not st.session_state.show_answer
        st.rerun()

    st.markdown("---")
    st.write("熟悉度：")

    # =========================
    # 核心：評分邏輯（已修正）
    # =========================
    def grade_card(level):
        idx = st.session_state["current_index"]
        schedule = st.session_state.schedule
        card_status = st.session_state.card_status

        st.session_state.review_count += 1

        status = card_status[idx]

        # 🔥 level 4 = 直接完成
        if level == 4:
            card_status[idx] = "done"
            schedule[idx] = 999999

        else:
            # 狀態變化
            if status == "new":
                card_status[idx] = "learning"

            elif status == "learning":
                if level == 3:
                    card_status[idx] = "review"

            elif status == "review":
                if level == 3:
                    card_status[idx] = "done"
                    schedule[idx] = 999999
                else:
                    card_status[idx] = "learning"

            # 間隔
            if card_status[idx] != "done":
                if level == 1:
                    schedule[idx] += 1
                elif level == 2:
                    schedule[idx] += 3
                elif level == 3:
                    schedule[idx] += 6

        st.session_state.schedule = schedule
        st.session_state.card_status = card_status
        st.session_state.show_answer = False

        st.rerun()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("1 完全不會（1張後）"):
            grade_card(1)

    with c2:
        if st.button("2 有點印象（3張後）"):
            grade_card(2)

    with c3:
        if st.button("3 大概會（6張後）"):
            grade_card(3)

    with c4:
        if st.button("4 很熟（不再出現）"):
            grade_card(4)

    st.markdown("---")

    with st.expander("熟悉度紀錄"):
        if st.session_state.levels:
            df = pd.DataFrame(
                [{"Card": k, "Level": v} for k, v in st.session_state.levels.items()]
            )
            st.dataframe(df)
        else:
            st.write("還沒有紀錄")