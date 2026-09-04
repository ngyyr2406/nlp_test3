# ============================================================
# STREAMLIT APP — NLP RESUME ANALYSIS SYSTEM
# ============================================================
# Frontend only.
#
# Streamlit
#     ↓
# Colab Flask API
#     ↓
# Existing NLP backend
#
# Pages:
# 1. Resume Database Search
# 2. Upload & Analyze Resume
#
# The backend remains responsible for:
# - SQLite database
# - Entity search
# - Semantic search
# - Entity aliases
# - model-best
# - PaddleOCR
# - GPT-OSS
# ============================================================

# IMPORTS
import streamlit as st
import requests
import json
import html
import base64
 
# STREAMLIT PAGE CONFIGURATION
st.set_page_config(
    page_title="NLP Resume Analysis System",
    page_icon="📄",
    layout="wide"
)
 
 
# CUSTOM CSS
st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #666666;
        margin-bottom: 25px;
    }

    .entity-box {
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 8px;
        border: 1px solid #dddddd;
    }

    .ocr-box {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        color: #111111;
        line-height: 1.7;
        white-space: normal;
        word-wrap: break-word;
    }

    .answer-box {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #cccccc;
        background-color: #f8f8f8;
        color: #111111;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)
 
# SESSION STATE
if "uploaded_results" not in st.session_state:
    st.session_state.uploaded_results = {}
 
if "api_url" not in st.session_state:
    st.session_state.api_url = ""
 
if "uploader_key_version" not in st.session_state:
    st.session_state.uploader_key_version = 0
 
if "current_resume_name" not in st.session_state:
    st.session_state.current_resume_name = ""
 
 
# SIDEBAR
st.sidebar.title("📄 NLP Resume Analysis")
 
st.sidebar.markdown(
    """
    ### Navigation
    """
)
 
page = st.sidebar.radio(
    "Select Page",
    [
        "Resume Database Search",
        "Upload & Analyze Resume"
    ]
)
 
 
# COLAB API URL
st.sidebar.markdown("---")
 
st.sidebar.subheader("Colab API")
 
api_url = st.sidebar.text_input(
    "API URL",
    value=st.session_state.api_url,
    placeholder="https://xxxx.ngrok-free.app"
)
 
api_url = api_url.strip().rstrip("/")
 
st.session_state.api_url = api_url
 
 
# HELPER — API URL CHECK
def check_api_url():
    if not api_url:
        st.error("Please enter your Colab API URL in the sidebar.")
        st.info("Copy the URL printed by your Colab ngrok cell.")
        return False
    return True
 

# HELPER — HEALTH CHECK
def get_health():
    try:
        response = requests.get(
            f"{api_url}/health",
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None
 
 
# SIDEBAR API STATUS
if api_url:
    health = get_health()
    if health:
        st.sidebar.success("Colab API connected")
    else:
        st.sidebar.error("Colab API unavailable")
 
# PAGE 1 — RESUME DATABASE SEARCH
if page == "Resume Database Search":
    st.markdown(
        '<div class="main-title">'
        '📄 Resume Database Search'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">'
        'Search the existing processed resume database using '
        'entity matching, alias matching and semantic search.'
        '</div>',
        unsafe_allow_html=True
    )
 
    # SEARCH INPUT
    st.subheader("🔎 Search Resume Database")
    query = st.text_input(
        "Enter an entity or natural-language question",
        placeholder=(
            "Example: Java "
            "or "
            "Which resume IDs have Java experience?"
        )
    )
 

    # FAST SEARCH TOGGLE (embeddings only, skips GPT-OSS)
    use_fast_search = st.checkbox(
        "⚡ Fast search (embeddings only)",
        value=False,
        help=(
            "When checked, calls /embedding-search directly: exact match "
            "→ alias match → MiniLM semantic search. Much faster since "
            "GPT-OSS is never loaded/run, but no natural-language answer "
            "is generated — only raw entity/resume matches with similarity "
            "scores."
        )
    )
 
    # SEARCH BUTTON
    search_clicked = st.button(
        "🔎 Search",
        type="primary",
        use_container_width=False
    )
 
 
    if search_clicked:
        if not check_api_url():
            st.stop()

        if not query.strip():
            st.warning("Please enter a search term or question.")
            st.stop()
 
 
        # CALL FLASK — /search OR /embedding-search
        endpoint = "/embedding-search" if use_fast_search else "/search"
 
        spinner_text = (
            "Running fast embedding search..."
            if use_fast_search
            else "Searching resume database..."
        )
 
        with st.spinner(spinner_text):
            try:
                if use_fast_search:
                    response = requests.post(
                        f"{api_url}{endpoint}",
                        json={
                            "query": query,
                            "max_documents": 5,
                            "top_k": 10
                        },
                        timeout=60
                    )
                else:
                    response = requests.post(
                        f"{api_url}{endpoint}",
                        json={
                            "query": query,
                            "max_documents": 5
                        },
                        timeout=300
                    )
                 
            except requests.exceptions.RequestException as e:
                st.error(
                    f"Could not connect to Colab API: {e}"
                )
                st.stop()
 
        # PROCESS RESPONSE
        if response.status_code != 200:
            try:
                error_data = response.json()
                st.error(
                    error_data.get(
                        "error",
                        "API search failed."
                    )
                )
 
            except Exception:
                st.error(
                    f"API returned status "
                    f"{response.status_code}."
                )
            st.stop()

        data = response.json()

        if not data.get("success", False):
            st.error(
                data.get(
                    "error",
                    "Search failed."
                )
            )
            st.stop()
 
 
        # NORMALISE RESULT SHAPE
        if use_fast_search:
            result = {
                "query_type": "ENTITY",
                "extracted_entity": data.get("query"),
                "matched_entity": data.get("matched_entity"),
                "similarity": data.get("similarity"),
                "matching_results": data.get("matching_results", 0),
                "similar_entities": data.get("similar_entities", []),
                "resume_ids": data.get("resume_ids", []),
                "records": [
                    {
                        "resume_id": record.get("resume_id"),
                        "entity_text": record.get("entity_text"),
                        "entity_label": record.get("entity_label"),
                        "similarity": record.get("similarity"),
                        "resume_content": record.get(
                            "highlighted_resume", ""
                        )
                    }
                    for record in data.get("records", [])
                ],
                "answer": None
            }
        else:
            result = data.get(
                "result",
                {}
            )
 

        # QUERY INFORMATION
        st.markdown("---")
        st.subheader("📊 Search Information")
        if use_fast_search:
            st.caption(
                "⚡ Fast search mode — exact match → alias match → "
                "MiniLM semantic search. No GPT-OSS answer generated."
            )
 
 
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric(
                "Query Type",
                result.get(
                    "query_type",
                    "N/A"
                )
            )
 
        with info_col2:
            st.metric(
                "Matching Results",
                result.get(
                    "matching_results",
                    0
                )
            )
 
        with info_col3:
            similarity = result.get(
                "similarity"
            )
            if similarity is not None:
                st.metric(
                    "Similarity",
                    f"{float(similarity):.4f}"
                )
            else:
                st.metric(
                    "Similarity",
                    "N/A"
                )
 
        # EXTRACTED ENTITY
        st.subheader("🏷️ Entity Extraction")
        extracted_entity = result.get("extracted_entity")
        matched_entity = result.get("matched_entity")
 
        col1, col2 = st.columns(2)
 
        with col1:
            st.write("**🔎 Extracted Entity**")
            if extracted_entity:
                st.info(str(extracted_entity))
            else:
                st.write("N/A")
 
        with col2:
            st.write("**🎯 Matching Entity**")
            if matched_entity:
                st.success(str(matched_entity))
            else:
                st.write("No matching entity")
 
        # SIMILAR ENTITIES
        similar_entities = result.get(
            "similar_entities",
            []
        )
 
        if similar_entities:
            with st.expander("🔗 View Similar Entities"):
                for entity in similar_entities:
                    entity_text = entity.get("entity_text","")
                    entity_label = entity.get("entity_label","")
                    entity_similarity = entity.get("similarity")
 
                    if entity_similarity is not None:
                        st.write(
                            f"**{entity_text}**  "
                            f"— {entity_label}  "
                            f"— similarity: "
                            f"{float(entity_similarity):.4f}"
                        )
                    else:
                        st.write(
                            f"**{entity_text}** "
                            f"— {entity_label}"
                        )
 
        # RESUME IDs
        resume_ids = result.get(
            "resume_ids",
            []
        )
        st.subheader("📁 Relevant Resume IDs")
 
        if resume_ids:
            st.write(
                ", ".join(
                    str(resume_id)
                    for resume_id in resume_ids
                )
            )
        else:
            st.warning("No matching resumes were found.")
 
        # DATABASE RECORDS
        records = result.get("records",[])
 
        st.subheader("📄 Relevant Resume Results")
 
        if not records:
            st.info("No resume records to display.")
         
        for index, record in enumerate(records,start=1):
            resume_id = record.get("resume_id","N/A")
            entity_text = record.get("entity_text","N/A")
            entity_label = record.get("entity_label","N/A")
            similarity = record.get("similarity")
            resume_content = record.get("resume_content","")
         
            # RESULT EXPANDER
            with st.expander(
                f"Resume {resume_id} — "
                f"{entity_text} "
                f"[{entity_label}]"
            ):
                result_col1, result_col2 = st.columns(2)
 
                with result_col1:
                    st.write("**Resume ID**")
                    st.write(str(resume_id))
                    st.write("**Entity**")
                    st.write(str(entity_text))
                    st.write("**Entity Label**")
                    st.write(str(entity_label))
 
                with result_col2:
                    st.write("**Similarity**")
 
                    if similarity is not None:
                        st.write(f"{float(similarity):.4f}")
                    else:
                        st.write("N/A")
 
                    if record.get(
                        "start_position"
                    ) is not None:
                        st.write("**Entity Position**")
                        st.write(
                            f"{record.get('start_position')} "
                            f"– "
                            f"{record.get('end_position')}"
                        )
                st.markdown("---")
                st.write("**Relevant Resume Content**")
 
                if resume_content:
                    st.text_area(
                        f"Resume content {index}",
                        value=str(
                            resume_content
                        ),
                        height=250,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                else:
                    st.info("No resume content available.")
 
        # GPT-OSS ANSWER
        answer = result.get("answer")
        st.subheader("🤖 GPT-OSS Answer")

        if answer:
            st.markdown(
                f"""
                <div class="answer-box">
                {html.escape(str(answer)).replace(chr(10), "<br>")}
                </div>
                """,
                unsafe_allow_html=True
            )
        elif use_fast_search:
            st.info(
                "Fast search mode does not call GPT-OSS. "
                "Uncheck '⚡ Fast search' to get a natural-language answer."
            )
        else:
            if result.get(
                "query_type"
            ) == "ENTITY":
                st.info(
                    "The input was classified as an ENTITY search. "
                    "No GPT-OSS database answer was generated."
                )
            else:
                st.info("No GPT-OSS answer was returned.")
 
# PAGE 2 — UPLOAD & ANALYZE RESUME (GROUPED MULTI-PAGE UPLOAD)
elif page == "Upload & Analyze Resume":
    st.markdown(
        '<div class="main-title">'
        '📤 Upload & Analyze Resume'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">'
        'Upload all page(s) belonging to ONE resume, give it a name, '
        'then submit. Start a new resume only after finishing the '
        'current one.'
        '</div>',
        unsafe_allow_html=True
    )
 
    # 1. RESUME NAME INPUT
    st.subheader("1️⃣ Name this resume")
    resume_name = st.text_input(
        "Resume name / candidate identifier",
        value=st.session_state.current_resume_name,
        placeholder="Example: John_Doe_Resume",
        key="resume_name_input"
    )
    st.session_state.current_resume_name = resume_name
 
    # 2. FILE UPLOADER — PAGES FOR THE CURRENT RESUME ONLY
    st.subheader("2️⃣ Upload all pages for this resume")
    uploader_key = f"resume_pages_{st.session_state.uploader_key_version}"
    uploaded_pages = st.file_uploader(
        "Choose file(s) — e.g. page 1 and page 2 of the same resume",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=uploader_key
    )
    if uploaded_pages:
        st.write(
            f"**{len(uploaded_pages)} page(s) staged for "
            f"'{resume_name or 'this resume'}':**"
        )
        for f in uploaded_pages:
            st.write(f"- {f.name}")
 
    # 3. SUBMIT THIS RESUME (all staged pages -> one API call)
    st.subheader("3️⃣ Submit this resume")
    submit_clicked = st.button(
        "✅ Analyze this resume",
        type="primary"
    )
 
    if submit_clicked:
        if not check_api_url():
            st.stop()
 
        if not uploaded_pages:
            st.warning("Please upload at least one page before submitting.")
            st.stop()
 
        final_resume_name = resume_name.strip() or uploaded_pages[0].name
 
        with st.spinner(
            f"Processing {len(uploaded_pages)} page(s) "
            f"for '{final_resume_name}'..."
        ):
            try:
                files_payload = [
                    (
                        "files",
                        (f.name, f.getvalue(), f.type)
                    )
                    for f in uploaded_pages
                ]
 
                response = requests.post(
                    f"{api_url}/analyze-resume-multi",
                    files=files_payload,
                    data={"resume_name": final_resume_name},
                    timeout=600
                )
 
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to Colab API: {e}")
                st.stop()
 
        if response.status_code != 200:
            try:
                error_data = response.json()
                st.error(error_data.get("error", "Analysis failed."))
            except Exception:
                st.error(f"API returned status {response.status_code}.")
            st.stop()
 
        data = response.json()
 
        if not data.get("success", False):
            st.error(data.get("error", "Analysis failed."))
            st.stop()
 
        # SAVE RESULT UNDER THE RESUME NAME (not per-file)
        st.session_state.uploaded_results[final_resume_name] = data
 
        # ALSO KEEP THE RAW PAGE BYTES FOR PREVIEW
        st.session_state.uploaded_results[final_resume_name]["_page_previews"] = [
            {
                "name": f.name,
                "bytes": f.getvalue(),
                "type": f.type
            }
            for f in uploaded_pages
        ]
 
        st.success(f"'{final_resume_name}' analyzed successfully.")
 
        # RESET FOR NEXT RESUME — bumping the key forces a fresh,
        # empty file_uploader widget on rerun
        st.session_state.uploader_key_version += 1
        st.session_state.current_resume_name = ""
        st.rerun()
 
    # DISPLAY ANALYZED RESUMES
    if st.session_state.uploaded_results:
        st.markdown("---")
        st.subheader("📂 Analyzed Resumes")
        for resume_name_key, result in st.session_state.uploaded_results.items():
            st.markdown("---")
            st.header(resume_name_key)
 
            # FILE INFORMATION
            file_col1, file_col2, file_col3 = st.columns(3)
            with file_col1:
                st.write("**Resume Name**")
                st.write(resume_name_key)
            with file_col2:
                confidence = result.get("mean_confidence", 0)
                st.write("**Average OCR Confidence**")
                st.write(f"{float(confidence):.4f}")
            with file_col3:
                st.write("**Pages**")
                st.write(str(result.get("page_count", 1)))
 
            # ORIGINAL UPLOADED PAGES
            st.subheader("📑 Uploaded Pages")
            page_previews = result.get("_page_previews", [])
            if page_previews:
                preview_cols = st.columns(min(len(page_previews), 4))
                for i, page in enumerate(page_previews):
                    col = preview_cols[i % len(preview_cols)]
                    with col:
                        ext = page["name"].lower().split(".")[-1]
                        if ext in {"png", "jpg", "jpeg", "webp"}:
                            st.image(
                                page["bytes"],
                                caption=page["name"],
                                use_container_width=True
                            )
                        else:
                            st.info(
                                f"{page['name']} "
                                f"(PDF page, converted for OCR)"
                            )
            else:
                st.info(
                    "Original page previews are not available "
                    "(loaded from a previous session)."
                )
 
            # OCR TEXT
            st.subheader("📝 OCR Extracted Text (combined across pages)")
            ocr_text = result.get("ocr_text", "")
            if ocr_text:
                st.text_area(
                    "OCR Text",
                    value=str(ocr_text),
                    height=300,
                    disabled=True,
                    label_visibility="collapsed"
                )
            else:
                st.warning("No OCR text was extracted.")
 
            # NER RESULTS
            st.subheader("🏷️ NER Results")
            entities = result.get("entities", [])
            if entities:
                ner_table = [
                    {
                        "Entity": entity.get("text", ""),
                        "Label": entity.get("label", ""),
                        "Start": entity.get("start", ""),
                        "End": entity.get("end", "")
                    }
                    for entity in entities
                ]
                st.dataframe(
                    ner_table,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No entities were detected.")
 
            # HIGHLIGHTED OCR
            st.subheader("✨ Highlighted Entities")
            highlighted_html = result.get("highlighted_html", "")
            if highlighted_html:
                st.markdown(
                    f"""
                    <div class="ocr-box">
                    {highlighted_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("No highlighted OCR available.")
 
            # RAW NER JSON
            with st.expander("🧾 View Raw NER JSON", expanded=False):
                raw_ner_json = result.get(
                    "ner_json",
                    {"filename": resume_name_key, "entities": entities}
                )
                st.json(raw_ner_json)
 
            # GPT-OSS Q&A
            st.subheader("🤖 GPT-OSS Q&A")
            question_key = "question_" + resume_name_key
            question = st.text_input(
                "Ask a question about this resume",
                placeholder="Example: Does this candidate have Java experience?",
                key=question_key
            )
 
            ask_button_key = "ask_" + resume_name_key
 
            if st.button("🤖 Ask GPT-OSS", key=ask_button_key):
                if not check_api_url():
                    st.stop()
                if not question.strip():
                    st.warning("Please enter a question.")
                elif not ocr_text.strip():
                    st.warning("No OCR text is available for this resume.")
                else:
                    with st.spinner(
                        "GPT-OSS is analyzing the uploaded resume..."
                    ):
                        try:
                            qa_response = requests.post(
                                f"{api_url}/uploaded-resume-question",
                                json={
                                    "question": question,
                                    "ocr_text": ocr_text
                                },
                                timeout=600
                            )
                        except requests.exceptions.RequestException as e:
                            st.error(f"Could not connect to Colab API: {e}")
                            qa_response = None
 
                    if qa_response is not None:
                        if qa_response.status_code != 200:
                            try:
                                error_data = qa_response.json()
                                st.error(
                                    error_data.get(
                                        "error",
                                        "GPT-OSS request failed."
                                    )
                                )
                            except Exception:
                                st.error(
                                    f"API returned status "
                                    f"{qa_response.status_code}."
                                )
                        else:
                            answer_data = qa_response.json()
                            if answer_data.get("success", False):
                                answer = answer_data.get("answer", "")
                                st.markdown(
                                    f"""
                                    <div class="answer-box">
                                    <strong>GPT-OSS Answer</strong>
                                    <br><br>
                                    {html.escape(str(answer)).replace(chr(10), "<br>")}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.error(
                                    answer_data.get(
                                        "error",
                                        "No answer returned."
                                    )
                                )
 
# FOOTER
st.sidebar.markdown("---")
st.sidebar.caption("NLP Resume Analysis System")
st.sidebar.caption("Streamlit Frontend → Colab Flask API")
