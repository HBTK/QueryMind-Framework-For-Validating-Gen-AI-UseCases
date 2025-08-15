import streamlit as st
import requests
import time
from pdf_report import generate_pdf_report

st.set_page_config(page_title="LLM Chatbot Validator", layout="wide")

st.title("LLM Chatbot Validator")
st.markdown("""
This application simulates a conversational AI that validates LLM-generated responses.
Your conversation history is maintained across turns and used to compute context similarity.
If you switch the use case, the session will be reset.
""")

st.sidebar.header("Configuration")
use_case = st.sidebar.selectbox("Select Use Case", 
                                ["Conversational", "Report Generation", "Summarization", "Paraphrasing", "Sentiment Analysis"],
                                key="use_case_selection")
# expected_sentiment = st.sidebar.selectbox("Expected Sentiment", ["neutral", "positive", "negative"], key="expected_sentiment")
expected_sentiment = "neutral"

if 'prev_use_case' not in st.session_state:
    st.session_state.prev_use_case = use_case
elif st.session_state.prev_use_case != use_case:
    st.session_state.conversation_history = []
    st.session_state.chat_log = []
    st.session_state.prev_use_case = use_case
    st.session_state.last_result = None
    if "pdf_bytes" in st.session_state:
        del st.session_state.pdf_bytes

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

st.subheader("Chat with the LLM")

for message in st.session_state.chat_log:
    role, text = message["role"], message["text"]
    if role == "user":
        st.markdown(f"<div style='text-align: right; background-color: #DCF8C6; padding: 8px; border-radius: 10px; margin-bottom: 4px;'><strong>User:</strong> {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; background-color: #FFFFFF; padding: 8px; border-radius: 10px; margin-bottom: 4px;'><strong>LLM:</strong> {text}</div>", unsafe_allow_html=True)

with st.form(key='chat_form'):
    user_query = st.text_input("Enter your message", key="user_query", label_visibility="visible")
    report_content = None
    original_text = None
    if use_case == "Report Generation":
        st.subheader("Report Content")
        report_content = st.text_area("Paste the full report content here", placeholder="Enter full report text...", height=250, label_visibility="visible")
    elif use_case in ["Summarization", "Paraphrasing"]:
        st.subheader("Original Text")
        original_text = st.text_area("Enter Original Text", placeholder="Enter the full article or source text...", height=250, label_visibility="visible")
    submitted = st.form_submit_button(label="Send")

if submitted:
    if user_query.strip() == "":
        st.warning("Please enter a message.")
    else:
        st.session_state.chat_log.append({"role": "user", "text": user_query})
        st.session_state.conversation_history.append(user_query)
        
        if use_case == "Report Generation":
            context_value = report_content if report_content else ""
        elif use_case in ["Summarization", "Paraphrasing"]:
            context_value = original_text if original_text else ""
        else:
            context_value = " ".join(st.session_state.conversation_history[-5:])
        
        payload = {
            "query": user_query,
            "conversation_history": st.session_state.conversation_history,
            "expected_sentiment": expected_sentiment,
            "report_content": report_content if use_case == "Report Generation" and report_content and report_content.strip() != "" else None,
            "original_text": original_text if use_case in ["Summarization", "Paraphrasing"] and original_text and original_text.strip() != "" else None,
            "summary": None
        }
        
        with st.spinner("Waiting for LLM response and validations..."):
            start_time = time.perf_counter()
            try:
                api_response = requests.post("http://127.0.0.1:8000/validate", json=payload)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                if api_response.status_code == 200:
                    result = api_response.json()
                    llm_response = result["llm_response"]
                    
                    st.session_state.chat_log.append({"role": "llm", "text": llm_response})
                    st.session_state.conversation_history.append(llm_response)
                    
                    st.markdown(f"<div style='text-align: left; background-color: #FFFFFF; padding: 8px; border-radius: 10px; margin-bottom: 4px;'><strong>LLM:</strong> {llm_response}</div>", unsafe_allow_html=True)
                    st.markdown(f"**Response Time:** {elapsed_time:.2f} seconds")
                    
                    st.subheader("Validation Metrics")
                    metrics = result["validator_scores"]
                    cols = st.columns(2)
                    for i, (key, value) in enumerate(metrics.items()):
                        if key == "context_similarity" and (value is None or value == 0):
                            continue
                        col = cols[i % 2]
                        if isinstance(value, (int, float)):
                            col.markdown(f"**{key.capitalize()} Score:** {value}")
                            col.progress(int(value * 100))
                        else:
                            col.markdown(f"**{key.capitalize()} Score:** {value}")
                    
                    st.markdown(f"**Average Score:** {result['average_score']}")
                    
                    perplexity_info = result.get("perplexity_validation", {})
                    if perplexity_info:
                        st.markdown(f"**Log-Normalized Perplexity Ratio:** {perplexity_info.get('log_normalized_perplexity_ratio', 'N/A')}")
                        st.markdown(f"**Perplexity Validation:** {'Valid' if perplexity_info.get('valid') else 'Not Valid'}")
                        st.markdown(f"**Perplexity Reason:** {perplexity_info.get('reason', '')}")
                    
                    if result["overall_valid"]:
                        st.success("Overall Validation: **Valid**")
                    else:
                        st.error("Overall Validation: **Not Valid**")
                    
                    st.session_state.last_result = {
                        "query": user_query,
                        "llm_response": llm_response,
                        "metrics": metrics,
                        "average_score": result["average_score"],
                        "overall_valid": result["overall_valid"],
                        "perplexity_info": perplexity_info,
                        "response_time": elapsed_time
                    }
                else:
                    st.error(f"API Error ({api_response.status_code}): {api_response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if st.button("Generate PDF Report"):
    if "last_result" not in st.session_state or not st.session_state.last_result:
        st.warning("No conversation available to generate a PDF report. Please interact with the chatbot first.")
    else:
        try:
            from pdf_report import generate_pdf_report
            res = st.session_state.last_result
            pdf_bytes = generate_pdf_report(
                query=res["query"],
                llm_response=res["llm_response"],
                metrics=res["metrics"],
                average_score=res["average_score"],
                overall_valid=res["overall_valid"],
                perplexity_info=res["perplexity_info"],
                response_time=res["response_time"]
            )
            st.session_state.pdf_bytes = pdf_bytes
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="llm_validation_report.pdf",
                mime="application/pdf"
            )
            st.success("PDF Report generated. Click the download button above to save it.")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
