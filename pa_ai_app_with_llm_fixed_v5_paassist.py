# ----------------- Findings Loader & Search -----------------
@st.cache_data(show_spinner=False)
def load_findings(uploaded=None):
    findings_df = pd.DataFrame()
    
    # 1. Try to load the pre-existing database file
    findings_db_path = "FindingsLibrary.csv"
    if os.path.exists(findings_db_path):
        try:
            findings_df = pd.read_csv(findings_db_path)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ FindingsLibrary.csv: {e}")
            findings_df = pd.DataFrame()

    # 2. If a new file is uploaded, combine it with the existing data
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                uploaded_df = pd.read_csv(uploaded)
            elif uploaded.name.endswith(('.xlsx', '.xls')):
                uploaded_df = pd.read_excel(uploaded, sheet_name=0) # Read the first sheet by default
            
            if not uploaded_df.empty:
                findings_df = pd.concat([findings_df, uploaded_df], ignore_index=True)
                st.success(f"อัปโหลดไฟล์ '{uploaded.name}' และรวมกับฐานข้อมูลเดิมแล้ว")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ที่อัปโหลด: {e}")
    
    # 3. Clean and return the combined dataframe
    if not findings_df.empty:
        for c in ["issue_title","issue_detail","cause_detail","recommendation","program","unit"]:
            if c in findings_df.columns:
                findings_df[c] = findings_df[c].fillna("")
        if "year" in findings_df.columns:
            findings_df["year"] = pd.to_numeric(findings_df["year"], errors="coerce").fillna(0).astype(int)
        if "severity" in findings_df.columns:
            findings_df["severity"] = pd.to_numeric(findings_df["severity"], errors="coerce").fillna(3).clip(1,5).astype(int)
    
    return findings_df


@st.cache_resource(show_spinner=False)
def build_tfidf_index(findings_df: pd.DataFrame):
    texts = (findings_df["issue_title"].fillna("") + " " +
             findings_df["issue_detail"].fillna("") + " " +
             findings_df["cause_detail"].fillna("") + " " +
             findings_df["recommendation"].fillna(""))
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return vec, X

def search_candidates(query_text, findings_df, vec, X, top_k=8):
    qv = vec.transform([query_text])
    sims = cosine_similarity(qv, X)[0]
    out = findings_df.copy()
    out["sim_score"] = sims
    # Normalize year for light recency boost
    if "year" in out.columns and out["year"].max() != out["year"].min():
        out["year_norm"] = (out["year"] - out["year"].min()) / (out["year"].max() - out["year"].min())
    else:
        out["year_norm"] = 0.0
    # Rerank
    out["sev_norm"] = out.get("severity", 3) / 5
    out["score"] = out["sim_score"]*0.65 + out["sev_norm"]*0.25 + out["year_norm"]*0.10
    cols = [
        "finding_id","year","unit","program","issue_title","issue_detail",
        "cause_category","cause_detail","recommendation","outcomes_impact","severity","score"
    ]
    cols = [c for c in cols if c in out.columns] + ["sim_score"]
    return out.sort_values("score", ascending=False).head(top_k)[cols]

# ----------------- App UI -----------------

# (The rest of your code from line 22 onwards, remains the same until the tab_issue section)

with tab_issue:
    st.subheader("🔎 แนะนำประเด็นตรวจจากรายงานเก่า (Issue Suggestions)")
    st.write("อัปโหลด FindingsLibrary.xlsx (ถ้าไม่มีจะอ่านจากไฟล์ FindingsLibrary.csv ที่มีอยู่เดิม)")
    uploaded = st.file_uploader("", type=["csv","xlsx"], key="findings_uploader")
    
    findings_df = load_findings(uploaded=uploaded)
    
    # The rest of the code in this block stays the same as your last provided version.
    # It will now work correctly with the combined data.
    if findings_df.empty:
        st.info("ไม่พบข้อมูล Findings ที่จะนำมาใช้ โปรดอัปโหลดไฟล์ หรือตรวจสอบว่ามีไฟล์ FindingsLibrary.csv อยู่ในโฟลเดอร์เดียวกัน")
    else:
        # The rest of your code for searching and generating suggestions...
        st.success(f"อัปโหลดข้อมูลสำเร็จ! รวมทั้งหมด {len(findings_df)} รายการ")
        vec, X = build_tfidf_index(findings_df)

        st.divider()

        st.write("กรอกข้อมูลที่เกี่ยวข้องกับงานที่ต้องการตรวจสอบ (จาก 6W2H, Logic Model, ฯลฯ)")
        query_text = st.text_area("ข้อความเกี่ยวกับเรื่องที่จะตรวจสอบ", height=150, help="เช่น โครงการพัฒนาแอพพลิเคชันเพื่อให้บริการประชาชน")

        if query_text:
            candidate_df = search_candidates(query_text, findings_df, vec, X)
            st.write("---")
            st.markdown("#### ข้อตรวจพบที่เกี่ยวข้องจากฐานข้อมูล")
            st.dataframe(candidate_df, use_container_width=True, hide_index=True)
            
            st.divider()
            st.markdown("#### 🤖 สร้างคำแนะนำประเด็นการตรวจสอบจาก AI")
            api_key = st.text_input("กรุณากรอก API Key เพื่อใช้บริการ AI:", type="password")

            if st.button("🚀 สร้างคำแนะนำจาก AI", type="primary"):
                if not api_key:
                    st.error("กรุณากรอก API Key ก่อนใช้งาน")
                else:
                    with st.spinner("กำลังประมวลผล..."):
                        try:
                            client = OpenAI(
                                api_key=api_key,
                                base_url="https://api.opentyphoon.ai/v1"
                            )
                            prompt = f"""
บทบาท: คุณคือผู้ช่วยนักตรวจสอบที่เชี่ยวชาญด้านการตรวจสอบผลการดำเนินงาน (Performance Audit)
ภารกิจ: จากข้อมูลที่กำหนดให้ (ประเด็นตรวจสอบ และ ข้อตรวจพบเก่าที่เกี่ยวข้อง)
- **แนะนำประเด็นการตรวจสอบ** ที่ควรให้ความสำคัญในปัจจุบัน โดยคำนึงถึงบริบทของการตรวจสอบผลการดำเนินงาน
- **คาดการณ์ข้อตรวจพบที่อาจจะเกิดขึ้น** พร้อมระดับโอกาส (สูง, กลาง, ต่ำ) โดยใช้ข้อตรวจพบเก่าเป็นแนวทาง
- **สรุปเป็นร่างรายงาน** (ประมาณ 100 คำ) โดยใช้ภาษาที่กระชับและตรงประเด็น

ข้อความจากผู้ใช้งาน (ประเด็นตรวจสอบ):
---
{query_text}
---

ข้อตรวจพบเก่าที่เกี่ยวข้อง (ใช้เป็นแนวทาง):
---
{candidate_df.to_string()}
---

รูปแบบที่ต้องการ:
##### ประเด็นการตรวจสอบที่ควรให้ความสำคัญ
[รายการประเด็น]

##### ข้อตรวจพบที่คาดว่าจะพบ (พร้อมระดับโอกาส)
[รายการข้อตรวจพบ]

##### ร่างรายงานตรวจสอบ (Preview)
[ข้อความสรุป]
"""

                            messages = [{"role": "user", "content": prompt}]
                            stream = client.chat.completions.create(
                                model="typhoon-v2.1-12b-instruct",
                                messages=messages,
                                stream=True,
                                temperature=0.7,
                            )
                            
                            full_text = ""
                            placeholder = st.empty()
                            for chunk in stream:
                                content = chunk.choices[0].delta.content or ""
                                full_text += content
                                placeholder.markdown(full_text)
                            
                            parts = full_text.split("#####")
                            issues_text = "#####" + parts[1] if len(parts) > 1 else ""
                            findings_text = "#####" + parts[2] if len(parts) > 2 else ""
                            report_text = "#####" + parts[3] if len(parts) > 3 else ""
                            
                            st.session_state["gen_issues"] = issues_text
                            st.session_state["gen_findings"] = findings_text
                            st.session_state["gen_report"] = report_text

                            st.success("สร้างคำแนะนำจาก AI เรียบร้อยแล้ว ✅")

                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ AI: {e}")
                            st.session_state["gen_issues"] = ""
                            st.session_state["gen_findings"] = ""
                            st.session_state["gen_report"] = ""
        
        st.markdown("<h4 style='color:blue;'>ประเด็นการตรวจสอบที่ควรให้ความสำคัญ</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:green; border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 200px; overflow-y: scroll;'>{st.session_state.get('gen_issues', '')}</div>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='color:blue;'>ข้อตรวจพบที่คาดว่าจะพบ (พร้อมระดับโอกาส)</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:green; border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 200px; overflow-y: scroll;'>{st.session_state.get('gen_findings', '')}</div>", unsafe_allow_html=True)

        st.markdown("<h4 style='color:blue;'>ร่างรายงานตรวจสอบ (Preview)</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:green; border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 400px; overflow-y: scroll;'>{st.session_state.get('gen_report', '')}</div>", unsafe_allow_html=True)
