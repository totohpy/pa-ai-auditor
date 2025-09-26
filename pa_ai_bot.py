# -*- coding: utf-8 -*-
import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
import glob
import pandas as pd
from io import StringIO

# กำหนดค่าหน้าเว็บ
st.set_page_config(page_title="🤖 PA Assistant", page_icon="🤖", layout="wide")

# กำหนดโฟลเดอร์เอกสารและข้อจำกัด
DOC_FOLDER = "Doc" 
MAX_CHARS_LIMIT = 100000

# ----------------- Utility Functions -----------------

def init_state():
    """Initializes only the necessary session state variables for the chatbot."""
    ss = st.session_state
    ss.setdefault("doc_context_local", "") # บริบทจากโฟลเดอร์ Doc
    ss.setdefault("doc_context_uploaded", "") # บริบทจากไฟล์ที่อัปโหลด
    
    # Initialize chat history
    ss.setdefault("chatbot_messages", [
        {"role": "assistant", "content": "สวัสดีครับ ผมคือผู้ช่วย AI อัจฉริยะ (PA Assistant) ผมพร้อมตอบคำถามโดยอ้างอิงจากเอกสารภายใน และข้อมูลจากเว็บไซต์ **www.audit.go.th** ครับ"}
    ])
    ss.setdefault("api_key_chatbot_standalone", st.secrets.get('OPENAI_API_KEY', ''))


def read_pdf_text_from_path(file_path):
    """Extracts text content from a PDF file using a local file path."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception:
        return ""

def read_pdf_text_from_uploaded(uploaded_file):
    """Extracts text content from an uploaded PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception:
        return ""
    
def get_text_from_file(file, source_type):
    """Handles text extraction based on file source (local path or uploaded object)."""
    file_name = file if source_type == 'local' else file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    current_text = ""
    
    try:
        if file_extension == '.pdf':
            if source_type == 'local':
                current_text = read_pdf_text_from_path(file)
            else:
                current_text = read_pdf_text_from_uploaded(file)
        elif file_extension == '.txt':
            if source_type == 'local':
                # ใช้ 'utf-8' สำหรับไฟล์ .txt
                with open(file, 'r', encoding='utf-8') as f:
                    current_text = f.read()
            else:
                # ใช้ 'utf-8' สำหรับไฟล์ที่อัปโหลด
                current_text = StringIO(file.getvalue().decode("utf-8")).read()
        elif file_extension == '.csv':
            if source_type == 'local':
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding='TIS-620')
            else:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding='TIS-620')

            current_text = df.to_string()
    
    except Exception as e:
        # st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file_name}: {e}")
        return "", ""

    return current_text, file_name


def process_documents(files, source_type, max_chars_limit, existing_context_len):
    """Processes a list of files (either local paths or uploaded objects) and returns context."""
    context = ""
    total_chars = existing_context_len
    
    for file in files:
        current_text, file_name = get_text_from_file(file, source_type)
        
        if current_text:
            # ตรวจสอบขีดจำกัด
            if total_chars + len(current_text) > max_chars_limit:
                remaining_chars = max_chars_limit - total_chars
                if remaining_chars > 0:
                    context += f"\n--- Start of Document: {file_name} (TRUNCATED) ---\n"
                    context += current_text[:remaining_chars] + f"\n... [ข้อความถูกตัดทอนจาก {file_name}]"
                st.warning(f"จำกัดจำนวนข้อความที่นำเข้าที่ {max_chars_limit:,} ตัวอักษร ข้อความที่เกินจะถูกตัดออก")
                break
            else:
                context += f"\n--- Start of Document: {file_name} ---\n"
                context += current_text
                context += f"\n--- End of Document: {file_name} ---\n"
                total_chars += len(current_text)
                if source_type == 'local':
                    st.toast(f"โหลด: {file_name} ({len(current_text):,} ตัวอักษร)")
                else:
                    st.toast(f"อัปโหลด: {file_name} ({len(current_text):,} ตัวอักษร)")

    return context, total_chars - existing_context_len # return เฉพาะส่วนที่เพิ่มใหม่

def load_local_documents_on_init():
    """Initial load function for Doc folder (run once)."""
    supported_files = []
    for ext in ['*.pdf', '*.txt', '*.csv']:
        supported_files.extend(glob.glob(os.path.join(DOC_FOLDER, ext)))
        
    if not supported_files:
        return ""

    context, total_chars_added = process_documents(supported_files, 'local', MAX_CHARS_LIMIT, 0)
    return context


# ----------------- Main App Logic -----------------

# 1. Initialize State และโหลดเอกสารเริ่มต้น
init_state()
if not st.session_state.doc_context_local and not st.session_state.get('local_load_attempted', False):
    # พยายามโหลดจาก Doc/ อัตโนมัติในครั้งแรก
    st.session_state.doc_context_local = load_local_documents_on_init()
    st.session_state.local_load_attempted = True

# *** เปลี่ยน Title ตามคำขอ ***
st.title("🤖 PA Assistant")

# 2. Sidebar for Configuration
with st.sidebar:
    st.subheader("🛠️ ตั้งค่าการใช้งาน")
    
    # API Key Input
    if not st.session_state.api_key_chatbot_standalone:
        st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
        st.session_state.api_key_chatbot_standalone = st.text_input(
            "กรุณากรอก OpenTyphoon API Key:",
            type="password",
            value=st.session_state.api_key_chatbot_standalone,
            key="api_key_input"
        )
    else:
        st.success("API Key ถูกโหลดแล้ว")

    st.divider()

    # สถานะบริบทจากโฟลเดอร์ Doc/
    local_chars = len(st.session_state.doc_context_local)
    if local_chars > 0:
        st.success(f"📂 **Doc/** : โหลดแล้ว {local_chars:,} ตัวอักษร")
    else:
        st.warning(f"📂 **Doc/** : ไม่พบเอกสารในโฟลเดอร์ '{DOC_FOLDER}'")

    st.subheader("⬆️ อัปโหลดเอกสารชั่วคราว (Ad-hoc)")
    uploaded_files = st.file_uploader(
        "อัปโหลดไฟล์ PDF, TXT, หรือ CSV (ใช้เฉพาะการสนทนานี้)",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True
    )
    
    if st.button("ประมวลผลเอกสารที่อัปโหลด", type="primary"):
        if uploaded_files:
            # รวมความยาวบริบทเดิม (จาก Doc/) เพื่อคำนวณขีดจำกัด
            existing_len = len(st.session_state.doc_context_local)
            uploaded_context, chars_added = process_documents(uploaded_files, 'uploaded', MAX_CHARS_LIMIT, existing_len)
            st.session_state.doc_context_uploaded = uploaded_context
            if chars_added > 0:
                 st.success(f"โหลดเอกสารที่อัปโหลดเสร็จสิ้น ({chars_added:,} ตัวอักษร)")
            else:
                 st.info("ไม่พบเอกสารที่ประมวลผลได้")

        else:
            st.session_state.doc_context_uploaded = ""
            st.info("ไม่มีไฟล์ที่อัปโหลด")

    uploaded_chars = len(st.session_state.doc_context_uploaded)
    if uploaded_chars > 0:
        st.info(f"💾 **Uploaded** : พร้อมใช้ {uploaded_chars:,} ตัวอักษร")
        if st.button("ล้างเอกสารที่อัปโหลด"):
             st.session_state.doc_context_uploaded = ""
             st.rerun()
    
    st.divider()

    # สรุปบริบทรวม
    total_chars = local_chars + uploaded_chars
    st.metric(label="บริบทรวมทั้งหมด (Max 100K)", value=f"{total_chars:,} ตัวอักษร")

    if st.button("🗑️ ล้างประวัติแชททั้งหมด"):
        st.session_state.chatbot_messages = [
             {"role": "assistant", "content": "สวัสดีครับ ผมคือผู้ช่วย AI อัจฉริยะ (PA Assistant) ผมพร้อมตอบคำถามโดยอ้างอิงจากเอกสารภายใน และข้อมูลจากเว็บไซต์ **www.audit.go.th** ครับ"}
        ]
        st.rerun()


# 3. Chat Display
for message in st.session_state.chatbot_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input and Processing
if prompt := st.chat_input("สอบถามเกี่ยวกับการตรวจสอบ หรือเอกสารที่โหลด/อัปโหลดไว้..."):

    # Check for API Key
    api_key = st.session_state.api_key_chatbot_standalone

    if not api_key:
        st.error("กรุณากรอก **API Key** ในแถบด้านข้าง (Sidebar) ก่อนใช้งาน")
        st.stop()
    
    # รวมบริบททั้งหมด: Doc/ + Uploaded (Uploaded จะถูกต่อท้าย)
    full_doc_context = st.session_state.doc_context_local + st.session_state.doc_context_uploaded
    doc_context = full_doc_context or "ไม่พบเอกสารภายใน"

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
                system_prompt = f"""
คุณคือผู้ช่วย AI อัจฉริยะ **(PA Assistant)** ที่เชี่ยวชาญการให้ข้อมูลเพื่อ **อ้างอิงการปฏิบัติงานสำหรับผู้ตรวจสอบของสำนักงานการตรวจเงินแผ่นดินภูมิภาคและจังหวัด** หน้าที่ของคุณคือตอบคำถามของผู้ใช้ให้ถูกต้องและครบถ้วนที่สุด โดยใช้แหล่งข้อมูลตามลำดับความสำคัญ:
1.  **เอกสารภายใน (Primary Source):** เนื้อหาจากโฟลเดอร์ "Doc" หรือที่ผู้ใช้เพิ่งอัปโหลด **จงยึดข้อมูลนี้เป็นหลัก**
2.  **ข้อมูลจากเว็บไซต์ สตง. (Secondary Source):** หากข้อมูลไม่พอหรือไม่ชัดเจน ให้อ้างอิงจากความรู้ที่คุณมีเกี่ยวกับเว็บไซต์ **https://www.audit.go.th/home** และข้อมูลที่เกี่ยวข้องกับการตรวจเงินแผ่นดินของไทย
3.  **ความรู้ทั่วไปจากอินเทอร์เน็ต (Tertiary Source):** ใช้เมื่อไม่พบคำตอบจากแหล่งข้อมูล 1 และ 2

**กฎการตอบ:**
- เมื่อตอบคำถาม ให้อ้างอิงเสมอว่าข้อมูลมาจากแหล่งใด (เช่น "จากเอกสาร [ชื่อไฟล์] ระบุว่า...", "จากข้อมูล สตง." หรือ "จากความรู้ทั่วไป") หากไม่ทราบชื่อไฟล์ให้บอกว่า "จากเอกสารที่ให้มา"
- หากข้อมูลในเอกสารภายในขัดแย้งกับข้อมูลภายนอก (เว็บไซต์ สตง. หรือทั่วไป) **ให้ยึดข้อมูลในเอกสารภายในเป็นหลัก**
- หากไม่พบคำตอบจากทุกแหล่งข้อมูล ให้ตอบว่า "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องทั้งในเอกสารและฐานข้อมูลของผม"

---
**บริบทจากเอกสารภายใน (Primary Source):**
{doc_context}
---

จากข้อมูลข้างต้นนี้ จงตอบคำถามล่าสุดของผู้ใช้
"""

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
