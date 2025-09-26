# -*- coding: utf-8 -*-
import streamlit as st
from openai import OpenAI
import os
import io
from PyPDF2 import PdfReader
from io import StringIO

# กำหนดค่าหน้าเว็บ
st.set_page_config(page_title="🤖 PA Chatbot - Standalone RAG", page_icon="🤖", layout="wide")


# ----------------- Utility Functions -----------------

def init_state():
    """Initializes only the necessary session state variables for the chatbot."""
    ss = st.session_state
    # Initialize chat history
    ss.setdefault("chatbot_messages", [
        {"role": "assistant", "content": "สวัสดีครับ ผมคือผู้ช่วยตรวจสอบ (PA Chatbot) ผมพร้อมตอบคำถามจากเอกสารที่ท่านอัปโหลดและข้อมูลบนอินเทอร์เน็ตแล้วครับ"}
    ])
    ss.setdefault("doc_context", "")
    # Use a specific key for the API key in this app
    ss.setdefault("api_key_chatbot_standalone", "") 


def read_pdf_text(uploaded_file):
    """Extracts text content from a PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่าน PDF: {e}")
        return ""


def handle_document_upload(uploaded_files, max_chars=100000):
    """Processes uploaded files and extracts text context, with character limit."""
    context = ""
    total_chars = 0
    
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        current_text = ""
        
        if file_extension == '.pdf':
            current_text = read_pdf_text(uploaded_file)
        elif file_extension in ['.txt', '.csv']:
            try:
                # To read file as string, use StringIO 
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                current_text = stringio.read()
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file_extension}: {e}")
                continue
        else:
            st.warning(f"ไฟล์นามสกุล '{file_extension}' ไม่รองรับการอ่าน")
            continue
            
        if current_text:
            if total_chars + len(current_text) > max_chars:
                # Truncate if adding the current document exceeds max_chars
                remaining_chars = max_chars - total_chars
                if remaining_chars > 0:
                    context += current_text[:remaining_chars] + f"\n... [ข้อความถูกตัดทอนจาก {uploaded_file.name}]"
                st.warning(f"จำกัดจำนวนข้อความที่นำเข้าที่ {max_chars} ตัวอักษร ข้อความที่เกินจะถูกตัดออก")
                break
            else:
                context += f"\n--- Start of Document: {uploaded_file.name} ---\n"
                context += current_text
                context += f"\n--- End of Document: {uploaded_file.name} ---\n"
                total_chars += len(current_text)
                st.success(f"โหลดเอกสาร '{uploaded_file.name}' ({len(current_text):,} ตัวอักษร)")

    return context


# ----------------- Main App Logic -----------------

# 1. Initialize State
init_state()

st.title("🤖 PA Chatbot - Standalone RAG")

# 2. Sidebar for Configuration
with st.sidebar:
    st.subheader("🛠️ ตั้งค่าการใช้งาน")
    st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
    st.session_state.api_key_chatbot_standalone = st.text_input(
        "กรุณากรอก API Key:", 
        type="password", 
        value=st.session_state.api_key_chatbot_standalone,
        key="api_key_input"
    )

    st.divider()
    
    st.subheader("📂 อัปโหลดเอกสาร (สำหรับบริบท)")
    uploaded_files = st.file_uploader(
        "อัปโหลดไฟล์ PDF, TXT, หรือ CSV", 
        type=["pdf", "txt", "csv"], 
        accept_multiple_files=True
    )
    
    if st.button("ประมวลผลเอกสาร", type="primary"):
        if uploaded_files:
            st.session_state.doc_context = handle_document_upload(uploaded_files)
        else:
            st.session_state.doc_context = ""
            st.info("ไม่มีไฟล์ถูกอัปโหลด")

    if st.session_state.doc_context:
        st.success(f"บริบทเอกสารพร้อมใช้: {len(st.session_state.doc_context):,} ตัวอักษร")
        if st.button("ล้างบริบทเอกสาร"):
             st.session_state.doc_context = ""
             st.rerun()
    else:
        st.info("ยังไม่มีบริบทเอกสาร (Chatbot จะใช้ความรู้ทั่วไป)")

    if st.button("ล้างประวัติแชท"):
        st.session_state.chatbot_messages = [
             {"role": "assistant", "content": "สวัสดีครับ ผมคือผู้ช่วยตรวจสอบ (PA Chatbot) ผมพร้อมตอบคำถามจากเอกสารที่ท่านอัปโหลดและข้อมูลบนอินเทอร์เน็ตแล้วครับ"}
        ]
        st.rerun()


# 3. Chat Display
for message in st.session_state.chatbot_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input and Processing
if prompt := st.chat_input("สอบถามเกี่ยวกับการตรวจสอบ PA หรือเอกสารที่อัปโหลด..."):
    
    # Check for API Key
    if not st.session_state.api_key_chatbot_standalone:
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
                    api_key=st.session_state.api_key_chatbot_standalone,
                    base_url="https://api.opentyphoon.ai/v1"
                )

                # 2. Prepare RAG prompt
                doc_context = st.session_state.doc_context or "ไม่พบเอกสารภายใน"

                system_prompt = f"""
คุณคือผู้ช่วยตรวจสอบ (Performance Audit Assistant) ที่เชี่ยวชาญด้านการตรวจสอบภาครัฐและผลสัมฤทธิ์
หน้าที่ของคุณคือการตอบคำถามของผู้ใช้โดยใช้ข้อมูลจาก 'บริบทจากเอกสารภายใน' เป็นหลัก หากไม่พบข้อมูลในเอกสาร ให้ใช้ความรู้ทั่วไปในการตอบ
- หากพบความขัดแย้งระหว่างเอกสารและความรู้ทั่วไป ให้ยึดข้อมูลในเอกสารเป็นหลักและอาจกล่าวถึงความขัดแย้งนั้น
- หากไม่พบคำตอบทั้งในเอกสารและความรู้ทั่วไป ให้ตอบว่า "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องทั้งในเอกสารและฐานข้อมูลของผม"

---
**บริบทจากเอกสารภายใน:**
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
