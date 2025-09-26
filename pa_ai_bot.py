# -*- coding: utf-8 -*-
import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
import glob
import pandas as pd
from io import StringIO

# กำหนดค่าหน้าเว็บ
st.set_page_config(page_title="🤖 PA Chatbot - Standalone RAG", page_icon="🤖", layout="wide")

# กำหนดโฟลเดอร์เอกสาร
DOC_FOLDER = "Doc" 
MAX_CHARS_LIMIT = 100000

# ----------------- Utility Functions -----------------

def init_state():
    """Initializes only the necessary session state variables for the chatbot."""
    ss = st.session_state
    # Initialize chat history
    ss.setdefault("chatbot_messages", [
        {"role": "assistant", "content": "สวัสดีครับ ผมคือผู้ช่วย AI อัจฉริยะ ผมพร้อมตอบคำถามจากเอกสารในโฟลเดอร์ **'Doc'** แล้วครับ"}
    ])
    ss.setdefault("doc_context", "")
    ss.setdefault("api_key_chatbot_standalone", st.secrets.get('OPENAI_API_KEY', ''))


def read_pdf_text(file_path):
    """Extracts text content from a PDF file using a local file path."""
    try:
        # ต้องเปิดไฟล์ในโหมดไบนารี (rb) เพื่อส่งให้ PdfReader
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่าน PDF: {e}")
        return ""


def load_local_documents(folder_path, max_chars=MAX_CHARS_LIMIT):
    """Reads all PDF, TXT, CSV files from a local folder and extracts text context."""
    context = ""
    total_chars = 0
    
    # ค้นหาไฟล์ที่รองรับในโฟลเดอร์
    supported_files = []
    for ext in ['*.pdf', '*.txt', '*.csv']:
        supported_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not supported_files:
        return "", f"ไม่พบไฟล์ที่รองรับ (.pdf, .txt, .csv) ในโฟลเดอร์ '{folder_path}'"

    for file_path in supported_files:
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1].lower()
        current_text = ""

        if file_extension == '.pdf':
            current_text = read_pdf_text(file_path)
        elif file_extension == '.txt':
            try:
                # อ่านไฟล์ TXT ด้วย encoding เป็น UTF-8
                with open(file_path, 'r', encoding='utf-8') as f:
                    current_text = f.read()
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file_name}: {e}")
                continue
        elif file_extension == '.csv':
            try:
                # อ่าน CSV ด้วย pandas และแปลงเป็น string เพื่อใช้เป็นบริบท
                df = pd.read_csv(file_path)
                current_text = df.to_string()
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV {file_name}: {e}")
                continue
        
        if current_text:
            # การจำกัดจำนวนตัวอักษรยังคงอยู่
            if total_chars + len(current_text) > max_chars:
                remaining_chars = max_chars - total_chars
                if remaining_chars > 0:
                    # เพิ่มชื่อไฟล์ในส่วนที่ถูกตัดทอน เพื่อให้ AI ทราบว่ามีการดึงข้อมูลมาจากไฟล์ใดบ้าง
                    context += f"\n--- Start of Document: {file_name} (TRUNCATED) ---\n"
                    context += current_text[:remaining_chars] + f"\n... [ข้อความถูกตัดทอนจาก {file_name}]"
                st.warning(f"จำกัดจำนวนข้อความที่นำเข้าที่ {max_chars:,} ตัวอักษร ข้อความที่เกินจะถูกตัดออก")
                break
            else:
                # เพิ่มชื่อไฟล์เข้าไปในบริบท
                context += f"\n--- Start of Document: {file_name} ---\n"
                context += current_text
                context += f"\n--- End of Document: {file_name} ---\n"
                total_chars += len(current_text)
                st.success(f"โหลดเอกสาร '{file_name}' ({len(current_text):,} ตัวอักษร)")

    return context, f"ประมวลผลเสร็จสิ้น: โหลดไป {total_chars:,} ตัวอักษร"


# ----------------- Main App Logic -----------------

# 1. Initialize State
init_state()

st.title("🤖 PA Chatbot - Standalone RAG")

# 2. Sidebar for Configuration
with st.sidebar:
    st.subheader("🛠️ ตั้งค่าการใช้งาน")
    
    # Only show API input if it's not set via Streamlit secrets
    if not st.session_state.api_key_chatbot_standalone:
        st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
        st.session_state.api_key_chatbot_standalone = st.text_input(
            "กรุณากรอก API Key:",
            type="password",
            value=st.session_state.api_key_chatbot_standalone,
            key="api_key_input"
        )
    else:
        st.success("API Key ถูกโหลดจาก Streamlit Secrets แล้ว")


    st.divider()

    st.subheader(f"📂 โหลดเอกสารจากโฟลเดอร์ '{DOC_FOLDER}'")
    st.info(f"ระบบจะอ่านไฟล์ **.pdf, .txt, .csv** ทั้งหมดในโฟลเดอร์ `{DOC_FOLDER}/`")

    if st.button("โหลดและประมวลผลเอกสาร", type="primary"):
        # เรียกใช้ฟังก์ชันใหม่เพื่อโหลดจากโฟลเดอร์
        st.session_state.doc_context, status_msg = load_local_documents(DOC_FOLDER)
        if not st.session_state.doc_context:
             st.warning(status_msg)
        else:
             st.info(status_msg)

    if st.session_state.doc_context:
        st.success(f"บริบทเอกสารพร้อมใช้: {len(st.session_state.doc_context):,} ตัวอักษร (จำกัดที่ {MAX_CHARS_LIMIT:,} ตัวอักษร)")
        if st.button("ล้างบริบทเอกสาร"):
             st.session_state.doc_context = ""
             st.rerun()
    else:
        st.info("ยังไม่มีบริบทเอกสาร (Chatbot จะใช้ความรู้ทั่วไป)")

    if st.button("ล้างประวัติแชท"):
        st.session_state.chatbot_messages = [
             {"role": "assistant", "content": "สวัสดีครับ ผมคือผู้ช่วย AI อัจฉริยะ ผมพร้อมตอบคำถามจากเอกสารในโฟลเดอร์ **'Doc'** แล้วครับ"}
        ]
        st.rerun()


# 3. Chat Display
for message in st.session_state.chatbot_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input and Processing
if prompt := st.chat_input("สอบถามเกี่ยวกับการตรวจสอบ PA หรือเอกสารที่โหลดไว้..."):

    # Check for API Key
    api_key = st.session_state.api_key_chatbot_standalone

    if not api_key:
        st.error("กรุณากรอก **API Key** ในแถบด้านข้าง (Sidebar) ก่อนใช้งาน")
        st.stop()

    # Display user message
    st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process assistant response
    with st.chat_message("assistant"):
        with st.spinner("กำลังประมวลผลคำตอบ..."):
            try:
                # 1. Prepare OpenAI Client
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.opentyphoon.ai/v1"
                )

                # 2. Prepare RAG prompt
                doc_context = st.session_state.doc_context or "ไม่พบเอกสารภายใน"

                # *** ส่วนที่ถูกแก้ไขตามคำขอของผู้ใช้ ***
                system_prompt = f"""
คุณคือผู้ช่วย AI อัจฉริยะ (Expert Assistant) หน้าที่ของคุณคือตอบคำถามของผู้ใช้ให้ถูกต้องและครบถ้วนที่สุด โดยใช้แหล่งข้อมูลสองแหล่ง:
1.  **ข้อมูลจากเอกสารภายใน (Primary Source):** นี่คือเนื้อหาที่ดึงมาจากไฟล์
 ในโฟลเดอร์ "Doc" ของระบบ จงยึดข้อมูลนี้เป็นหลักในการตอบคำถามเสมอ
2.  **ความรู้ทั่วไปและข้อมูลจากอินเทอร์เน็ต (Secondary Source):** หากคำตอบไม่มีอยู่ในเอกสารภายใน ให้ใช้ความรู้ที่คุณมีจากการฝึกฝน (ซึ่งเทียบเท่าการค้นหาข้อมูลบนอินเทอร์เน็ต) เพื่อตอบคำถาม

**กฎการตอบ:**
- เมื่อตอบคำถาม ให้อ้างอิงเสมอว่าข้อมูลมาจากแหล่งใด (เช่น "จากเอกสาร [ชื่อไฟล์] ระบุว่า..." หรือ "จากเอกสารที่ให้มา" ระบุว่า...) หากไม่ทราบชื่อไฟล์ให้บอกว่า "จากเอกสารที่ให้มา"
- หากข้อมูลในเอกสารขัดแย้งกับข้อมูลทั่วไป ให้ยึดข้อมูลในเอกสารเป็นหลักและอาจกล่าวถึงความขัดแย้งนั้น
- หากไม่พบคำตอบทั้งในเอกสารและความรู้ทั่วไป ให้ตอบว่า "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องทั้งในเอกสารและฐานข้อมูลของผม"

---
**บริบทจากเอกสารภายใน:**
{doc_context}
---

จากข้อมูลข้างต้นนี้ จงตอบคำถามล่าสุดของผู้ใช้
"""
                # *** สิ้นสุดส่วนที่ถูกแก้ไข ***

                messages_for_api = [
                    {"role": "system", "content": system_prompt}
                ]
                # Add chat history, but keep it concise (last 10 messages)
                for msg in st.session_state.chatbot_messages[-10:]:
                    messages_for_api.append(msg)

                # 3. Call LLM API
                response_stream = client.chat.completions.create(
                    model="typhoon-v2.1-12b-instruct",
                    messages=messages_for_api,
                    temperature=0.5,
                    max_tokens=3072,
                    stream=True
                )

                # 4. Write stream response
                response = st.write_stream(response_stream)
                st.session_state.chatbot_messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_message = f"เกิดข้อผิดพลาด: {e}"
                st.error(error_message)
                st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
