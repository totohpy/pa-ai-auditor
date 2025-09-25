# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import xlsxwriter

st.set_page_config(page_title="Planning Studio (+ Issue Suggestions)", page_icon="🧭", layout="wide")

# ----------------- Session Init -----------------
def init_state():
    ss = st.session_state
    ss.setdefault("plan", {
        "plan_id": "PLN-" + datetime.now().strftime("%y%m%d-%H%M%S"),
        "plan_title": "",
        "program_name": "",
        "who": "", "what": "", "where": "", "when": "", "why": "", "how": "", "how_much": "", "whom": "",
        "objectives": "", "scope": "", "assumptions": "", "status": "Draft"
    })
    ss.setdefault("logic_items", pd.DataFrame(columns=["item_id","plan_id","type","description","metric","unit","target","source"]))
    ss.setdefault("methods", pd.DataFrame(columns=["method_id","plan_id","type","tool_ref","sampling","questions","linked_issue","data_source","frequency"]))
    ss.setdefault("kpis", pd.DataFrame(columns=["kpi_id","plan_id","level","name","formula","numerator","denominator","unit","baseline","target","frequency","data_source","quality_requirements"]))
    ss.setdefault("risks", pd.DataFrame(columns=["risk_id","plan_id","description","category","likelihood","impact","mitigation","hypothesis"]))
    ss.setdefault("audit_issues", pd.DataFrame(columns=["issue_id","plan_id","title","rationale","linked_kpi","proposed_methods","source_finding_id","issue_detail", "recommendation"]))
    ss.setdefault("gen_issues", "")
    ss.setdefault("gen_findings", "")
    ss.setdefault("gen_report", "")

def next_id(prefix, df, col):
    if df.empty: return f"{prefix}-001"
    nums = []
    for x in df[col]:
        try:
            nums.append(int(str(x).split("-")[-1]))
        except:
            pass
    n = max(nums) + 1 if nums else 1
    return f"{prefix}-{n:03d}"

def df_download_link(df: pd.DataFrame, filename: str, label: str):
    buf = BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="text/csv")

def create_excel_template():
    """Generates an Excel template file with two sheets: 'Data' and 'คำอธิบาย'."""
    cols = [
        "finding_id","report_id","year","unit","program",
        "issue_title","issue_detail","cause_category","cause_detail",
        "evidence_type","recommendation","outcomes_affected","kpi_touchpoints","severity"
    ]
    data_df = pd.DataFrame(columns=cols)
    
    # Create the "คำอธิบาย" sheet
    descriptions = {
        "finding_id": "รหัสของข้อตรวจพบ",
        "report_id": "รหัสรายงาน",
        "year": "ปีที่รายงาน",
        "unit": "หน่วยงานที่ถูกตรวจสอบ",
        "program": "ชื่อโครงการ/โปรแกรม",
        "issue_title": "ชื่อประเด็นการตรวจสอบ",
        "issue_detail": "รายละเอียดของประเด็น",
        "cause_category": "หมวดหมู่ของสาเหตุ",
        "cause_detail": "รายละเอียดของสาเหตุ",
        "evidence_type": "ประเภทของหลักฐาน",
        "recommendation": "ข้อเสนอแนะ",
        "outcomes_affected": "ผลกระทบที่อาจเกิดขึ้น",
        "kpi_touchpoints": "KPI ที่เกี่ยวข้อง",
        "severity": "ระดับความรุนแรงของข้อตรวจพบ (1-5)"
    }
    desc_df = pd.DataFrame(list(descriptions.items()), columns=["Column Name", "Description"])

    # Save to BytesIO object
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data_df.to_excel(writer, sheet_name='Data', index=False)
        desc_df.to_excel(writer, sheet_name='คำอธิบาย', index=False)
    
    # Return to the beginning of the stream
    output.seek(0)
    return output

# ----------------- Findings Loader & Search -----------------
@st.cache_data(show_spinner=False)
def load_findings(default_path="FindingsLibrary.csv", uploaded=None):
    if uploaded is not None:
        file_extension = os.path.splitext(uploaded.name)[1].lower()
        if file_extension in [".xls", ".xlsx"]:
            # แก้ไข: ระบุ sheet_name="Data" เพื่อให้ pandas อ่านเฉพาะชีตข้อมูล
            try:
                df = pd.read_excel(uploaded, sheet_name="Data")
            except ValueError:
                st.error("ไม่พบชีต 'Data' ในไฟล์ Excel ที่อัปโหลด กรุณาตรวจสอบไฟล์แม่แบบ")
                df = pd.DataFrame()
        elif file_extension == ".csv":
            df = pd.read_csv(uploaded)
        else:
            st.warning("รูปแบบไฟล์ไม่ถูกต้อง กรุณาอัปโหลดไฟล์ CSV หรือ Excel")
            df = pd.DataFrame()
    else:
        try:
            # แก้ไข: ลองอ่านไฟล์ csv เป็นค่าเริ่มต้น
            df = pd.read_csv(default_path)
        except Exception as e:
            st.warning("ไม่พบไฟล์ FindingsLibrary.csv โปรดอัปโหลดไฟล์")
            df = pd.DataFrame()
    # Clean nulls
    for c in ["issue_title","issue_detail","cause_detail","recommendation","program","unit"]:
        if c in df.columns:
            df[c] = df[c].fillna("")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    if "severity" in df.columns:
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(3).clip(1,5).astype(int)
    return df

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
        "cause_category","cause_detail","recommendation","outcomes_affected","severity","score"
    ]
    cols = [c for c in cols if c in out.columns] + ["sim_score"]
    return out.sort_values("score", ascending=False).head(top_k)[cols]

# ----------------- App UI -----------------
init_state()
plan = st.session_state["plan"]
logic_df = st.session_state["logic_items"]
methods_df = st.session_state["methods"]
kpis_df = st.session_state["kpis"]
risks_df = st.session_state["risks"]
audit_issues_df = st.session_state["audit_issues"]

st.title("🧭 Planning Studio – Performance Audit (แนะนำประเด็นการตรวจสอบ)")

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist = st.tabs([
    "1) แผน & 6W2H", "2) Logic Model", "3) Methods", "4) KPIs", "5) Risks", "6) Issue Suggestions", "7) Preview/Export", "✨ PA Audit Assist ✨"
])

with tab_plan:
    st.subheader("ข้อมูลแผน (Plan) กรุณาระบุข้อมูล")
    with st.container(border=True):
        c1, c2, c3 = st.columns([2,2,1])
        with c1:
            plan["plan_title"] = st.text_input("ชื่อแผน/เรื่องที่จะตรวจ", plan["plan_title"])
            plan["program_name"] = st.text_input("ชื่อโครงการ/แผนงาน", plan["program_name"])
            plan["objectives"] = st.text_area("วัตถุประสงค์การตรวจ", plan["objectives"])
        with c2:
            plan["scope"] = st.text_area("ขอบเขตการตรวจ", plan["scope"])
            plan["assumptions"] = st.text_area("สมมุติฐาน/ข้อจำกัดข้อมูล", plan["assumptions"])
        with c3:
            st.text_input("Plan ID", plan["plan_id"], disabled=True)
            plan["status"] = st.selectbox("สถานะ", ["Draft","Published"], index=0)

    st.divider()
    st.subheader("6W2H กรุณาระบุข้อมูล")

    # ----- New Section for LLM Generation (text area) -----
    with st.container(border=True):
        st.markdown("##### 🚀 สร้าง 6W2H อัตโนมัติด้วย AI")
        st.write("คัดลอกและวางข้อความจากไฟล์ของคุณในช่องด้านล่างนี้")
        uploaded_text = st.text_area("วางข้อความเกี่ยวกับเรื่องที่จะตรวจสอบ ที่ต้องการให้ AI ช่วยสรุป 6W2H", height=200, key="uploaded_text")
        st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
        api_key_6w2h = st.text_input("กรุณากรอก API Key เพื่อใช้บริการ AI:", type="password", key="api_key_6w2h")

        if st.button("🚀 สร้าง 6W2H จากข้อความ", type="primary"):
            if not uploaded_text:
                st.error("กรุณาวางข้อความในช่องก่อน")
            elif not api_key_6w2h:
                st.error("กรุณากรอก API Key ก่อนใช้งาน")
            else:
                with st.spinner("กำลังประมวลผล..."):
                    try:
                        user_prompt = f"""
จากข้อความด้านล่างนี้ กรุณาสรุปและแยกแยะข้อมูลให้เป็น 6W2H ได้แก่ Who, Whom, What, Where, When, Why, How, และ How much โดยให้อยู่ในรูปแบบ key-value ที่ชัดเจน
ข้อความ:
---
{uploaded_text}
---
รูปแบบที่ต้องการ:
Who: [ข้อความ]
Whom: [ข้อความ]
What: [ข้อความ]
Where: [ข้อความ]
When: [ข้อความ]
Why: [ข้อความ]
How: [ข้อความ]
How Much: [ข้อความ]
"""
                        client = OpenAI(
                            api_key=api_key_6w2h,
                            base_url="https://api.opentyphoon.ai/v1"
                        )
                        response = client.chat.completions.create(
                            model="typhoon-v2.1-12b-instruct",
                            messages=[{"role": "user", "content": user_prompt}],
                            temperature=0.7,
                            max_tokens=1024,
                        )
                        llm_output = response.choices[0].message.content
                        
                        # Debugging section: Show raw LLM output
                        with st.expander("แสดงผลลัพธ์จาก AI"):
                            st.write(llm_output)

                        # Parse the LLM output with more flexibility and update session state
                        lines = llm_output.strip().split('\n')
                        for line in lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                normalized_key = key.strip().lower().replace(' ', '_')
                                value = value.strip()
                                # Use a more robust mapping
                                if normalized_key == 'how_much':
                                    st.session_state.plan['how_much'] = value
                                elif normalized_key == 'whom':
                                    st.session_state.plan['whom'] = value
                                elif normalized_key == 'who':
                                    st.session_state.plan['who'] = value
                                elif normalized_key == 'what':
                                    st.session_state.plan['what'] = value
                                elif normalized_key == 'where':
                                    st.session_state.plan['where'] = value
                                elif normalized_key == 'when':
                                    st.session_state.plan['when'] = value
                                elif normalized_key == 'why':
                                    st.session_state.plan['why'] = value
                                elif normalized_key == 'how':
                                    st.session_state.plan['how'] = value

                        st.success("สร้าง 6W2H เรียบร้อยแล้ว! กรุณาตรวจสอบข้อมูลแล้วคัดลอกตามรายละเอียดด้านล่าง")
                        st.balloons()
                        # st.rerun() is no longer needed here as widgets will get updated automatically
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ AI: {e}")
    
    # ----------------------------------------------
    
    with st.container(border=True):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.session_state.plan["who"] = st.text_input("Who (ใคร)", value=st.session_state.plan["who"], key="who_input")
            st.session_state.plan["whom"] = st.text_input("Whom (ถึงใคร)", value=st.session_state.plan["whom"], key="whom_input")
            st.session_state.plan["what"] = st.text_input("What (ทำอะไร)", value=st.session_state.plan["what"], key="what_input")
            st.session_state.plan["where"] = st.text_input("Where (ที่ไหน)", value=st.session_state.plan["where"], key="where_input")
        with cc2:
            st.session_state.plan["when"] = st.text_input("When (เมื่อใด)", value=st.session_state.plan["when"], key="when_input")
            st.session_state.plan["why"] = st.text_area("Why (ทำไม)", value=st.session_state.plan["why"], key="why_input")
        with cc3:
            st.session_state.plan["how"] = st.text_area("How (อย่างไร)", value=st.session_state.plan["how"], key="how_input")
            st.session_state.plan["how_much"] = st.text_input("How much (เท่าไร)", value=st.session_state.plan["how_much"], key="how_much_input")

# Note: The st.session_state.plan variables are already being updated by the widgets themselves due to the key.
# A separate loop is not required anymore.

with tab_logic:
    st.subheader("ระบุข้อมูล Logic Model: Input → Activities → Output → Outcome → Impact")
    st.dataframe(logic_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่มรายการใน Logic Model"):
        with st.container(border=True):
            colA, colB, colC = st.columns(3)
            with colA:
                typ = st.selectbox("ประเภท", ["Input","Activity","Output","Outcome","Impact"])
                desc = st.text_input("คำอธิบาย/รายละเอียด")
                metric = st.text_input("ตัวชี้วัด/metric (เช่น จำนวน, สัดส่วน)")
            with colB:
                unit = st.text_input("หน่วย", value="หน่วย", key="logic_unit")
                target = st.text_input("เป้าหมาย", value="", key="logic_target")
            with colC:
                source = st.text_input("แหล่งข้อมูล", value="", key="logic_source")
                if st.button("เพิ่ม Logic Item", type="primary"):
                    new_row = pd.DataFrame([{
                        "item_id": next_id("LG", logic_df, "item_id"),
                        "plan_id": plan["plan_id"],
                        "type": typ, "description": desc, "metric": metric,
                        "unit": unit, "target": target, "source": source
                    }])
                    st.session_state["logic_items"] = pd.concat([logic_df, new_row], ignore_index=True)
                    st.rerun()

with tab_method:
    st.subheader("ระบุวิธีการเก็บข้อมูล (Methods)")
    st.dataframe(methods_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่ม Method"):
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                mtype = st.selectbox("ชนิด", ["observe","interview","questionnaire","document"])
                tool_ref = st.text_input("รหัส/อ้างอิงเครื่องมือ", value="")
                sampling = st.text_input("วิธีคัดเลือกตัวอย่าง", value="")
            with c2:
                questions = st.text_area("คำถาม/ประเด็นหลัก")
                linked_issue = st.text_input("โยงประเด็นตรวจ", value="")
            with c3:
                data_source = st.text_input("แหล่งข้อมูล", value="", key="method_data_source")
                frequency = st.text_input("ความถี่", value="ครั้งเดียว", key="method_frequency")
                if st.button("เพิ่ม Method", type="primary"):
                    new_row = pd.DataFrame([{
                        "method_id": next_id("MT", methods_df, "method_id"),
                        "plan_id": plan["plan_id"],
                        "type": mtype, "tool_ref": tool_ref, "sampling": sampling,
                        "questions": questions, "linked_issue": linked_issue,
                        "data_source": data_source, "frequency": frequency
                    }])
                    st.session_state["methods"] = pd.concat([methods_df, new_row], ignore_index=True)
                    st.rerun()

with tab_kpi:
    st.subheader("ระบุตัวชี้วัด (KPIs)")
    st.dataframe(kpis_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่ม KPI เอง"):
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                level = st.selectbox("ระดับ", ["output","outcome"])
                name = st.text_input("ชื่อ KPI")
                formula = st.text_input("สูตร/นิยาม")
            with col2:
                numerator = st.text_input("ตัวตั้ง (numerator)")
                denominator = st.text_input("ตัวหาร (denominator)")
                unit = st.text_input("หน่วย", value="%", key="kpi_unit")
            with col3:
                baseline = st.text_input("Baseline", value="")
                target = st.text_input("Target", value="")
                freq = st.text_input("ความถี่", value="รายไตรมาส")
                data_src = st.text_input("แหล่งข้อมูล", value="", key="kpi_data_source")
                quality = st.text_input("ข้อกำหนดคุณภาพข้อมูล", value="ถูกต้อง/ทันเวลา", key="kpi_quality")
                if st.button("เพิ่ม KPI", type="primary"):
                    new_row = pd.DataFrame([{
                        "kpi_id": next_id("KPI", kpis_df, "kpi_id"),
                        "plan_id": plan["plan_id"],
                        "level": level, "name": name, "formula": formula,
                        "numerator": numerator, "denominator": denominator, "unit": unit,
                        "baseline": baseline, "target": target, "frequency": freq,
                        "data_source": data_src, "quality_requirements": quality
                    }])
                    st.session_state["kpis"] = pd.concat([kpis_df, new_row], ignore_index=True)
                    st.rerun()

with tab_risk:
    st.subheader("ระบุความเสี่ยง (Risks)")
    st.dataframe(risks_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่ม Risk"):
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                desc = st.text_area("คำอธิบายความเสี่ยง")
                category = st.selectbox("หมวด", ["policy","org","data","process","people"])
            with c2:
                likelihood = st.select_slider("โอกาสเกิด (1-5)", options=[1,2,3,4,5], value=3)
                impact = st.select_slider("ผลกระทบ (1-5)", options=[1,2,3,4,5], value=3)
            with c3:
                mitigation = st.text_area("มาตรการลดความเสี่ยง")
                hypothesis = st.text_input("สมมุติฐานที่ต้องทดสอบ")
                if st.button("เพิ่ม Risk", type="primary"):
                    new_row = pd.DataFrame([{
                        "risk_id": next_id("RSK", risks_df, "risk_id"),
                        "plan_id": plan["plan_id"],
                        "description": desc, "category": category,
                        "likelihood": likelihood, "impact": impact,
                        "mitigation": mitigation, "hypothesis": hypothesis
                    }])
                    st.session_state["risks"] = pd.concat([risks_df, new_row], ignore_index=True)
                    st.rerun()

with tab_issue:
    st.subheader("🔎 แนะนำประเด็นตรวจจากรายงานเก่า (Issue Suggestions)")
    with st.container(border=True):
        st.download_button(
            label="⬇️ ดาวน์โหลดไฟล์แม่แบบ FindingsLibrary.xlsx",
            data=create_excel_template(),
            file_name="FindingsLibrary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        uploaded = st.file_uploader("อัปโหลด FindingsLibrary.csv หรือ .xlsx (ถ้าไม่มีจะอ่านจากฐานข้อมูลที่มีอยู่)", type=["csv", "xlsx", "xls"])
        findings_df = load_findings(uploaded=uploaded)
        if findings_df.empty:
            st.stop()
        vec, X = build_tfidf_index(findings_df)

    seed = f"""
Who:{plan.get('who','')} What:{plan.get('what','')} Where:{plan.get('where','')}
When:{plan.get('when','')} Why:{plan.get('why','')} How:{plan.get('how','')}
Outputs:{' | '.join(logic_df[logic_df['type']=='Output']['description'].tolist())}
Outcomes:{' | '.join(logic_df[logic_df['type']=='Outcome']['description'].tolist())}
"""
    st.text_area("สรุปบริบท (แก้ไขได้):", seed, height=140)
    if st.button("ค้นหาประเด็นที่ใกล้เคียง", type="primary"):
        results = search_candidates(seed, findings_df, vec, X, top_k=8)
        st.session_state["issue_results"] = results
        st.success(f"พบประเด็นที่เกี่ยวข้อง {len(results)} รายการ")
        
    results = st.session_state.get("issue_results", pd.DataFrame())
    
    if not results.empty:
        st.divider()
        st.subheader("ผลลัพธ์การค้นหา")
        for i, row in results.reset_index(drop=True).iterrows():
            with st.container(border=True):
                title_txt = row["issue_title"] if "issue_title" in row else "(ไม่มีชื่อประเด็น)"
                unit_txt = row["unit"] if "unit" in row else "-"
                prog_txt = row["program"] if "program" in row else "-"
                year_txt = int(row["year"]) if "year" in row and str(row["year"]).isdigit() else row.get("year", "-")
                st.markdown(f"**{title_txt}** \nหน่วย: {unit_txt} • โครงการ: {prog_txt} • ปี {year_txt}")
                cause_cat = row["cause_category"] if "cause_category" in row else "-"
                cause_detail = row["cause_detail"] if "cause_detail" in row else "-"
                st.caption(f"สาเหตุ: *{cause_cat}* — {cause_detail}")

                with st.expander("รายละเอียด/ข้อเสนอแนะ (เดิม)"):
                    st.write(row["issue_detail"] if "issue_detail" in row else "-")
                    st.caption("ข้อเสนอแนะเดิม: " + (row["recommendation"] if "recommendation" in row and row["recommendation"] else "-"))
                    impact = row["outcomes_affected"] if "outcomes_affected" in row else "-"
                    sim = row["sim_score"] if "sim_score" in row else 0
                    score = row["score"] if "score" in row else 0
                    
                    st.markdown(f"**ผลกระทบที่อาจเกิดขึ้น:** {impact}  •  <span style='color:red;'>**คะแนนความเกี่ยวข้อง**</span>: {score:.3f} (<span style='color:blue;'>**Similarity Score**</span>={sim:.3f})", unsafe_allow_html=True)
                    st.caption("💡 **คำอธิบาย:** **คะแนนความเกี่ยวข้อง** (ยิ่งสูงยิ่งดี) = ความคล้ายคลึงของข้อความ + ความรุนแรงของปัญหา + ความใหม่ของข้อมูล")
                    st.caption("**Similarity Score** คือค่าความคล้ายคลึงระหว่างข้อความในแผนงานของคุณกับรายงานเก่า (0.000 - 1.000)")

                c1, c2 = st.columns([3,1])
                with c1:
                    default_rat = f"อ้างอิงกรณีเดิม ปี {year_txt} | หน่วย: {unit_txt}"
                    st.text_area("เหตุผลที่ควรตรวจ (สำหรับแผนนี้)", key=f"rat_{i}", value=default_rat)
                    st.text_input("KPI ที่เกี่ยว (ถ้ามี)", key=f"kpi_{i}")
                    st.text_input("วิธีเก็บข้อมูลที่เสนอ", key=f"mth_{i}", value="สัมภาษณ์/สังเกต/ตรวจเอกสาร")

                with c2:
                    if st.button("➕ เพิ่มเข้าแผน", key=f"add_{i}", type="secondary"):
                        rationale_val = st.session_state.get(f"rat_{i}", "")
                        linked_kpi_val = st.session_state.get(f"kpi_{i}", "")
                        proposed_methods_val = st.session_state.get(f"mth_{i}", "")
                        issue_detail_val = row.get("issue_detail", "")
                        recommendation_val = row.get("recommendation", "")

                        cols = ["issue_id","plan_id","title","rationale","linked_kpi","proposed_methods","source_finding_id","issue_detail","recommendation"]
                        if "audit_issues" not in st.session_state or not isinstance(st.session_state["audit_issues"], pd.DataFrame):
                            st.session_state["audit_issues"] = pd.DataFrame(columns=cols)
                        for c in cols:
                            if c not in st.session_state["audit_issues"].columns:
                                st.session_state["audit_issues"][c] = pd.Series(dtype="object")
                        
                        curr = st.session_state["audit_issues"]
                        nums = []
                        if not curr.empty and "issue_id" in curr.columns:
                            for x in curr["issue_id"].astype(str):
                                if x.startswith("ISS-"):
                                    try:
                                        nums.append(int(x.split("-")[-1]))
                                    except Exception:
                                        pass
                        new_id = f"ISS-{(max(nums) if nums else 0)+1:03d}"

                        title_val = title_txt
                        finding_id = row["finding_id"] if "finding_id" in row else ""

                        new = pd.DataFrame([{
                            "issue_id": new_id,
                            "plan_id": plan.get("plan_id",""),
                            "title": title_val,
                            "rationale": rationale_val,
                            "linked_kpi": linked_kpi_val,
                            "proposed_methods": proposed_methods_val,
                            "source_finding_id": finding_id,
                            "issue_detail": issue_detail_val,
                            "recommendation": recommendation_val
                        }])

                        st.session_state["audit_issues"] = pd.concat([st.session_state["audit_issues"], new], ignore_index=True)
                        st.success("เพิ่มประเด็นเข้าแผนแล้ว ✅")
                        st.rerun()
    if not st.session_state.get("issue_results", pd.DataFrame()).empty:
        st.divider()
    st.markdown("### ประเด็นที่ถูกเพิ่มเข้าแผน")
    st.dataframe(st.session_state["audit_issues"], use_container_width=True, hide_index=True)


with tab_preview:
    st.subheader("สรุปแผน (Preview)")
    with st.container(border=True):
        st.markdown(f"**Plan ID:** {plan['plan_id']}  \n**ชื่อแผน:** {plan['plan_title']}  \n**โครงการ:** {plan['program_name']}  \n**สถานะ:** {plan['status']}")
    st.markdown("### บทนำ (จาก 6W2H)")
    with st.container(border=True):
        intro = f"""
- **Who**: {plan['who']}
- **Whom**: {plan['whom']}
- **What**: {plan['what']}
- **Where**: {plan['where']}
- **When**: {plan['when']}
- **Why**: {plan['why']}
- **How**: {plan['how']}
- **How much**: {plan['how_much']}
"""
        st.markdown(intro)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Logic Model")
        st.dataframe(st.session_state["logic_items"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["logic_items"], "logic_items.csv", "⬇️ ดาวน์โหลด Logic Items (CSV)")
    with c2:
        st.markdown("### Methods")
        st.dataframe(st.session_state["methods"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["methods"], "methods.csv", "⬇️ ดาวน์โหลด Methods (CSV)")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### KPIs")
        st.dataframe(st.session_state["kpis"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["kpis"], "kpis.csv", "⬇️ ดาวน์โหลด KPIs (CSV)")
    with c4:
        st.markdown("### Risks")
        st.dataframe(st.session_state["risks"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["risks"], "risks.csv", "⬇️ ดาวน์โหลด Risks (CSV)")

    st.markdown("### Audit Issues ที่เพิ่มเข้ามา")
    if not st.session_state["audit_issues"].empty:
        display_issues_df = st.session_state["audit_issues"].copy()
        display_issues_df = display_issues_df.rename(columns={
            "issue_id": "รหัสประเด็น",
            "title": "ชื่อประเด็น",
            "rationale": "เหตุผลที่ควรตรวจ",
            "issue_detail": "รายละเอียด",
            "recommendation": "ข้อเสนอแนะ"
        })
        display_cols = ["รหัสประเด็น", "ชื่อประเด็น", "เหตุผลที่ควรตรวจ", "รายละเอียด", "ข้อเสนอแนะ"]
        st.dataframe(display_issues_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("ยังไม่มีประเด็นการตรวจสอบที่เพิ่มเข้ามาในแผน")

    if not st.session_state["audit_issues"].empty:
        df_download_link(st.session_state["audit_issues"], "audit_issues.csv", "⬇️ ดาวน์โหลด Audit Issues (CSV)")

    st.divider()
    plan_df = pd.DataFrame([plan])
    df_download_link(plan_df, "plan.csv", "⬇️ ดาวน์โหลด Plan (CSV)")
    st.success("พร้อมเชื่อม Glide / Sheets ต่อได้ทันที")

with tab_assist:
    st.subheader("💡 PA Audit Assist (ขับเคลื่อนด้วย LLM)")
    st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
    api_key = st.text_input("กรุณากรอก API Key เพื่อใช้บริการ AI:", type="password")

    if st.button("🚀 สร้างคำแนะนำจาก AI", type="primary"):
        if not api_key:
            st.error("กรุณากรอก API Key ก่อนใช้งาน")
        else:
            with st.spinner("กำลังสร้างคำแนะนำ..."):
                try:
                    issues_for_llm = st.session_state['audit_issues'][['title', 'rationale']]
                    plan_summary = f"""
วัตถุประสงค์: {plan['objectives']}
ขอบเขต: {plan['scope']}
สมมุติฐาน/ข้อจำกัด: {plan['assumptions']}
---
6W2H:
ใคร (Who): {plan['who']}
ถึงใคร (Whom): {plan['whom']}
ทำอะไร (What): {plan['what']}
ที่ไหน (Where): {plan['where']}
เมื่อใด (When): {plan['when']}
ทำไม (Why): {plan['why']}
อย่างไร (How): {plan['how']}
เท่าไร (How much): {plan['how_much']}
---
Logic Model:
{st.session_state['logic_items'].to_string()}
---
ประเด็นที่เพิ่มจากรายงานเก่า:
{issues_for_llm.to_string()}
"""
                    user_prompt = f"""
จากข้อมูลแผนการตรวจสอบด้านล่างนี้ กรุณาช่วยสร้างคำแนะนำ 3 อย่าง ได้แก่
1. ประเด็นการตรวจสอบที่ควรให้ความสำคัญ
2. ข้อตรวจพบที่คาดว่าจะพบ (พร้อมระบุระดับโอกาสที่จะเจอ: สูง/กลาง/ต่ำ)
3. ร่างรายงานตรวจสอบที่จะเจอ
---
{plan_summary}
---
กรุณาสร้างคำตอบตามรูปแบบด้านล่างนี้เท่านั้น:
<ประเด็นการตรวจสอบที่ควรให้ความสำคัญ>
[ข้อความสำหรับส่วนที่ 1]
</ประเด็นการตรวจสอบที่ควรให้ความสำคัญ>

<ข้อตรวจพบที่คาดว่าจะพบ>
[ข้อความสำหรับส่วนที่ 2]
</ข้อตรวจพบที่คาดว่าจะพบ>

<ร่างรายงานตรวจสอบที่จะเจอ>
[ข้อความสำหรับส่วนที่ 3]
</ร่างรายงานตรวจสอบที่จะเจอ>
"""

                    client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.opentyphoon.ai/v1"
                    )
                    
                    messages = [
                        {"role": "system", "content": "การตรวจสอบผลสัมฤทธิ์และประสิทธิภาพการดำเนินงาน (Performance Audit)"},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    response = client.chat.completions.create(
                        model="typhoon-v2.1-12b-instruct",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2048,
                        top_p=0.9,
                    )

                    full_response = response.choices[0].message.content

                    issue_start = full_response.find("<ประเด็นการตรวจสอบที่ควรให้ความสำคัญ>") + len("<ประเด็นการตรวจสอบที่ควรให้ความสำคัญ>")
                    issue_end = full_response.find("</ประเด็นการตรวจสอบที่ควรให้ความสำคัญ>")
                    issues_text = full_response[issue_start:issue_end].strip()
                    
                    finding_start = full_response.find("<ข้อตรวจพบที่คาดว่าจะพบ>") + len("<ข้อตรวจพบที่คาดว่าจะพบ>")
                    finding_end = full_response.find("</ข้อตรวจพบที่คาดว่าจะพบ>")
                    findings_text = full_response[finding_start:finding_end].strip()

                    report_start = full_response.find("<ร่างรายงานตรวจสอบที่จะเจอ>") + len("<ร่างรายงานตรวจสอบที่จะเจอ>")
                    report_end = full_response.find("</ร่างรายงานตรวจสอบที่จะเจอ>")
                    report_text = full_response[report_start:report_end].strip()

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
