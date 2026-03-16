"""
Customer Service – Complaint & Issue Management — Streamlit UI
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator import PipelineOrchestrator

INPUTS_DIR = PROJECT_ROOT / "inputs" / "samples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(
    page_title="Customer Service – Complaint & Issue Management",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""<style>
    .status-pass { color: #2e7d32; }
    .status-fail { color: #c62828; }
    div[data-testid="stExpander"] { border: 1px solid #ddd; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
</style>""", unsafe_allow_html=True)


# --------------- helpers ---------------

def load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_all_pipeline_runs() -> list:
    run_dir = OUTPUTS_DIR / "pipeline_run"
    if not run_dir.exists():
        return []
    runs = []
    for f in sorted(run_dir.glob("*.json"), reverse=True):
        data = load_json(f)
        if data:
            data["_file"] = f.name
            runs.append(data)
    return runs


def get_sample_list() -> list:
    if not INPUTS_DIR.exists():
        return []
    return sorted([d.name for d in INPUTS_DIR.iterdir() if d.is_dir()])


def load_sample_inputs(sample_name: str) -> dict:
    sample_dir = INPUTS_DIR / sample_name
    result = {}
    ct = sample_dir / "complaint_text.txt"
    if ct.exists():
        result["complaint_text"] = ct.read_text(encoding="utf-8")
    cp = sample_dir / "customer_profile.json"
    if cp.exists():
        result["customer_profile"] = load_json(cp)
    ih = sample_dir / "interaction_history.json"
    if ih.exists():
        result["interaction_history"] = load_json(ih)
    return result


def run_pipeline_sync(complaint_path, profile_path, history_path):
    orch = PipelineOrchestrator()
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            orch.run(complaint_path, profile_path, history_path)
        )
    finally:
        loop.close()


# --------------- header + navigation ---------------

st.title("Customer Service – Complaint & Issue Management")
st.caption("6-Agent Resolution Pipeline  ·  Autogen 0.4  ·  SAP Integration")

tab_dashboard, tab_run, tab_explorer, tab_data, tab_audit = st.tabs([
    "Dashboard", "Run Pipeline", "Agent Explorer", "Data & Config", "Audit Trail"
])

# --------------- tab: Dashboard ---------------

with tab_dashboard:
    st.subheader("Overview")
    st.write("End-to-end complaint resolution using a 6-agent pipeline.")

    # Architecture
    st.subheader("Pipeline Flow")
    st.markdown("""
| Step | Agent | Role |
|------|-------|------|
| 1 | Complaint Classifier | Categorize complaint, assign severity & priority |
| 2 | Evidence Collector | Pull SAP data (sales orders, deliveries, billing, warehouse) |
| 3 | Remedy Planner | Build resolution plan with cost estimate |
| 4 | Credit Trigger | Approve / reject credit; generate SAP FI posting |
| 5 | Customer Communicator | Draft response (email / SMS / portal) |
| 6 | Audit Logger | Validate all outputs, SOX & GDPR compliance check |
""")

    runs = load_all_pipeline_runs()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pipeline Runs", len(runs))
    with col2:
        completed = sum(1 for r in runs if r.get("pipeline_run", {}).get("status") == "completed")
        st.metric("Successful", completed)
    with col3:
        if runs:
            avg_time = sum(r.get("pipeline_run", {}).get("elapsed_seconds", 0) for r in runs) / len(runs)
            st.metric("Avg Time", f"{avg_time:.1f}s")
        else:
            st.metric("Avg Time", "—")
    with col4:
        st.metric("Sample Scenarios", len(get_sample_list()))

    if runs:
        st.subheader("Recent Runs")
        rows = []
        for r in runs[:5]:
            pr = r.get("pipeline_run", {})
            res = r.get("results", {})
            cat = res.get("agent1_classifier", {}).get("complaint_category", res.get("agent1_classifier", {}))
            credit = res.get("agent4_credit", {}).get("approval", {})
            credit_display = "N/A"
            if credit.get("status") == "approved":
                credit_display = f"₹{credit.get('amount', 0):,.0f}"
            rows.append({
                "Complaint ID": pr.get("complaint_id", "—"),
                "Category": cat.get("category", "—"),
                "Status": pr.get("status", "—"),
                "Credit": credit_display,
                "Time (s)": round(pr.get("elapsed_seconds", 0), 2),
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# --------------- tab: Run Pipeline ---------------

with tab_run:
    st.subheader("Run Pipeline")
    st.write("Upload the 3 input files and run the 6-agent pipeline.")

    uploaded_files = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        f1 = st.file_uploader("Complaint Text (.txt)", type=["txt"], key="up_complaint")
    with col2:
        f2 = st.file_uploader("Customer Profile (.json)", type=["json"], key="up_profile")
    with col3:
        f3 = st.file_uploader("Interaction History (.json)", type=["json"], key="up_history")

    if f1 and f2 and f3:
        uploaded_files = {"complaint": f1, "profile": f2, "history": f3}
        st.success("All 3 files uploaded — ready to run!")

    st.markdown("---")

    can_run = bool(uploaded_files)
    if st.button("Run Full Pipeline", type="primary", disabled=not can_run, width='stretch'):

        try:
            import tempfile
            tmp = Path(tempfile.mkdtemp())
            (tmp / "complaint_text.txt").write_bytes(uploaded_files["complaint"].getvalue())
            (tmp / "customer_profile.json").write_bytes(uploaded_files["profile"].getvalue())
            (tmp / "interaction_history.json").write_bytes(uploaded_files["history"].getvalue())
            complaint_path = tmp / "complaint_text.txt"
            profile_path = tmp / "customer_profile.json"
            history_path = tmp / "interaction_history.json"
        except Exception as exc:
            st.error(f"Failed to process uploaded files: {exc}")
            st.stop()

        # Run with progress
        agent_names = [
            "Agent 1: Complaint Classifier",
            "Agent 2: Evidence Collector",
            "Agent 3: Remedy Planner",
            "Agent 4: Credit Trigger",
            "Agent 5: Customer Communicator",
            "Agent 6: Audit Logger",
        ]

        progress_bar = st.progress(0, text="Initializing pipeline...")

        start = time.time()

        for i, name in enumerate(agent_names):
            progress_bar.progress(i / 6, text=f"Running {name}...")
            time.sleep(0.1)

        try:
            with st.spinner("Processing..."):
                result = run_pipeline_sync(complaint_path, profile_path, history_path)

            elapsed = time.time() - start
            progress_bar.progress(1.0, text=f"Done — {elapsed:.1f}s")
        except FileNotFoundError as exc:
            progress_bar.empty()
            st.error(f"Input file not found: {exc}")
            st.stop()
        except json.JSONDecodeError as exc:
            progress_bar.empty()
            st.error(f"Invalid JSON in uploaded file: {exc}")
            st.stop()
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Pipeline failed: {exc}")
            st.stop()

        # Store result in session state
        st.session_state["last_result"] = result
        st.session_state["last_elapsed"] = elapsed

        st.markdown("---")
        st.subheader("Results")

        res = result.get("results", {})
        pr = result.get("pipeline_run", {})

        # Top metrics
        a1 = res.get("agent1_classifier", {})
        cat_data = a1.get("complaint_category", a1)
        a4 = res.get("agent4_credit", {})
        credit = a4.get("approval", {})

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Category", cat_data.get("category", "—"))
        with m2:
            st.metric("Severity", cat_data.get("severity_level", "—"))
        with m3:
            credit_val = credit.get("amount", "—")
            if credit.get("status") == "not_applicable":
                credit_val = "N/A"
            st.metric("Credit Amount", f"₹{credit_val}" if isinstance(credit_val, (int, float)) else credit_val)
        with m4:
            st.metric("Processing Time", f"{elapsed:.2f}s")

        agent_keys = [
            ("agent1_classifier", "1. Complaint Classifier"),
            ("agent2_evidence", "2. Evidence Collector"),
            ("agent3_remedy", "3. Remedy Planner"),
            ("agent4_credit", "4. Credit Trigger"),
            ("agent5_communicator", "5. Customer Communicator"),
            ("agent6_audit", "6. Audit Logger"),
        ]

        for key, label in agent_keys:
            agent_data = res.get(key, {})
            with st.expander(label, expanded=(key in ("agent1_classifier", "agent4_credit"))):
                if key == "agent1_classifier":
                    cc = agent_data.get("complaint_category", agent_data)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f"**Category:** `{cc.get('category', '—')}`")
                    c2.markdown(f"**Severity:** `{cc.get('severity_level', '—')}`")
                    c3.markdown(f"**Priority:** `{cc.get('priority_tag', '—')}`")
                    c4.markdown(f"**Accuracy:** {cc.get('accuracy_score', '—')}")
                    st.markdown(f"**Routing Queue:** `{cc.get('routing_queue', '—')}`")
                    st.markdown(f"**Validation:** `{cc.get('validation_status', '—')}`")

                elif key == "agent2_evidence":
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**Complete:** {'✅' if agent_data.get('completeness_flag') else '❌'}")
                    c2.markdown(f"**Validation:** `{agent_data.get('validation_status', '—')}`")
                    c3.markdown(f"**Accuracy:** {agent_data.get('accuracy_score', '—')}")
                    summary = agent_data.get("summary", {})
                    if summary:
                        st.markdown(f"**Issue:** {summary.get('issue', '—')}")
                        st.markdown(f"**Status:** {summary.get('current_status', '—')}")
                    so = agent_data.get("sales_order")
                    if so:
                        st.markdown(f"**Sales Order:** `{so.get('VBAK', {}).get('VBELN', '—')}` — {len(so.get('VBAP', []))} line items")
                    bl = agent_data.get("billing")
                    if bl:
                        total = sum(i.get("NETWR", 0) for i in bl.get("VBRP", []))
                        st.markdown(f"**Billing:** `{bl.get('VBRK', {}).get('VBELN', '—')}` — ₹{total:,.2f}")

                elif key == "agent3_remedy":
                    actions = agent_data.get("actions", [])
                    cost = agent_data.get("cost_estimate", {})
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**Actions:** {len(actions)}")
                    c2.markdown(f"**Cost:** ₹{cost.get('amount', 0):,.2f} {cost.get('currency', 'INR')}")
                    c3.markdown(f"**Compliant:** {'✅' if agent_data.get('policy_compliance') else '❌'}")
                    if actions:
                        action_rows = []
                        for a in actions:
                            det = a.get("details", {})
                            action_rows.append({
                                "Type": a.get("type", "—"),
                                "Amount": f"₹{det.get('estimated_amount', '—')}" if det.get("estimated_amount") else "—",
                                "Details": det.get("action", det.get("carrier", det.get("team", "—"))),
                            })
                        st.dataframe(pd.DataFrame(action_rows), width='stretch', hide_index=True)

                elif key == "agent4_credit":
                    appr = agent_data.get("approval", {})
                    status = appr.get("status", "—")
                    status_icon = "✅" if status == "approved" else ("ℹ️" if status == "not_applicable" else "❌")
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**Status:** {status_icon} `{status}`")
                    c2.markdown(f"**Amount:** ₹{appr.get('amount', 0):,.2f}" if appr.get("amount") else f"**Reason:** {appr.get('reason', '—')}")
                    c3.markdown(f"**Credit Doc:** `{appr.get('credit_doc', '—')}`")
                    if appr.get("credit_types"):
                        st.markdown(f"**Credit Types:** {', '.join(appr['credit_types'])}")
                    sap = agent_data.get("sap_fi_posting", {})
                    if sap:
                        st.markdown(f"**SAP FI Doc:** `{sap.get('BKPF', {}).get('BELNR', '—')}`")
                        st.markdown(f"**Posting Text:** {sap.get('BSEG', {}).get('SGTXT', '—')}")

                elif key == "agent5_communicator":
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**Channel:** `{agent_data.get('dispatch_channel', agent_data.get('channel', '—'))}`")
                    comp = agent_data.get("compliance", {})
                    c2.markdown(f"**GDPR:** {'✅' if comp.get('gdpr') else '❌'}")
                    c3.markdown(f"**Brand:** {'✅' if comp.get('brand') else '❌'}")
                    to = agent_data.get("to", {})
                    if to:
                        st.markdown(f"**To:** {to.get('name', '—')} ({to.get('email', '—')})")
                    body = agent_data.get("body", "") or agent_data.get("email_body", "")
                    if body:
                        st.text_area("Message Preview", body, height=120, disabled=True, label_visibility="collapsed")

                elif key == "agent6_audit":
                    summary = agent_data.get("audit_summary", {})
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f"**Entries:** {summary.get('total_entries', '—')}")
                    c2.markdown(f"**Valid:** {summary.get('valid_entries', '—')}")
                    c3.markdown(f"**Compliant:** {summary.get('compliant_entries', '—')}")
                    c4.markdown(f"**All Steps:** {'✅' if summary.get('all_steps_completed') else '❌'}")
                    trail = agent_data.get("audit_trail_entries", [])
                    if trail:
                        st.dataframe(pd.DataFrame(trail), width='stretch', hide_index=True)

                with st.popover("Raw JSON"):
                    st.json(agent_data)


# --------------- tab: Agent Explorer ---------------

with tab_explorer:
    st.subheader("Agent Explorer")
    st.write("Inspect individual agent outputs from past pipeline runs.")

    runs = load_all_pipeline_runs()
    if not runs:
        st.warning("No pipeline runs found. Go to 'Run Pipeline' to execute one first.")
    else:
        run_options = {}
        for r in runs:
            pr = r.get("pipeline_run", {})
            cid = pr.get("complaint_id", r.get("_file", "unknown"))
            cat = r.get("results", {}).get("agent1_classifier", {}).get("complaint_category", {}).get("category", "?")
            label = f"{cid} — {cat} ({pr.get('elapsed_seconds', '?')}s)"
            run_options[label] = r

        selected_label = st.selectbox("Select a pipeline run", list(run_options.keys()))
        run_data = run_options[selected_label]
        res = run_data.get("results", {})

        agent_map = {
            "Agent 1: Classifier": "agent1_classifier",
            "Agent 2: Evidence": "agent2_evidence",
            "Agent 3: Remedy": "agent3_remedy",
            "Agent 4: Credit": "agent4_credit",
            "Agent 5: Communicator": "agent5_communicator",
            "Agent 6: Audit": "agent6_audit",
        }

        selected_agent = st.selectbox("Select Agent", list(agent_map.keys()))
        agent_key = agent_map[selected_agent]
        agent_data = res.get(agent_key, {})

        if not agent_data:
            st.warning(f"No data for {selected_agent} in this run.")
        else:
            st.markdown(f"### {selected_agent} — Detailed Output")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### Structured View")
                if agent_key == "agent1_classifier":
                    cc = agent_data.get("complaint_category", agent_data)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Category", cc.get("category", "—"))
                    m2.metric("Severity", cc.get("severity_level", "—"))
                    m3.metric("Priority", cc.get("priority_tag", "—"))
                elif agent_key == "agent3_remedy":
                    actions = agent_data.get("actions", [])
                    cost = agent_data.get("cost_estimate", {})
                    m1, m2 = st.columns(2)
                    m1.metric("Actions", len(actions))
                    m2.metric("Total Cost", f"₹{cost.get('amount', 0):,.2f}")
                    if actions:
                        st.dataframe(pd.DataFrame(actions), width='stretch', hide_index=True)
                elif agent_key == "agent4_credit":
                    appr = agent_data.get("approval", {})
                    m1, m2 = st.columns(2)
                    m1.metric("Status", appr.get("status", "—"))
                    m2.metric("Amount", f"₹{appr.get('amount', 0):,.2f}" if isinstance(appr.get("amount"), (int, float)) else "—")

            with col2:
                st.markdown("#### Metadata")
                st.markdown(f"**Agent ID:** `{agent_data.get('agent_id', '—')}`")
                st.markdown(f"**Mode:** `{agent_data.get('mode', '—')}`")
                st.markdown(f"**Validation:** `{agent_data.get('validation_status', '—')}`")
                ts = agent_data.get("timestamp", "—")
                st.markdown(f"**Timestamp:** {ts}")

            st.markdown("---")
            st.markdown("#### Raw JSON")
            st.json(agent_data)


# --------------- tab: Data & Config ---------------

with tab_data:
    st.subheader("Data & Configuration")
    st.write("Browse SAP data sources, policies, and agent configurations.")

    data_tab1, data_tab2, data_tab3 = st.tabs(["SAP Data", "Policies", "Configuration"])

    with data_tab1:
        st.markdown("### SAP Data Files")
        data_files = sorted(DATA_DIR.glob("*.json")) if DATA_DIR.exists() else []
        if not data_files:
            st.info("No data files found.")
        else:
            selected_data = st.selectbox("Select Data File", [f.name for f in data_files], key="data_select")
            data = load_json(DATA_DIR / selected_data)
            if isinstance(data, list):
                st.markdown(f"**Records:** {len(data)}")
                for i, record in enumerate(data):
                    with st.expander(f"Record {i+1}", expanded=(i == 0)):
                        st.json(record)
            else:
                st.json(data)

    with data_tab2:
        st.markdown("### Policy Files")
        policy_files = sorted(CONFIG_DIR.glob("*_policy.json")) if CONFIG_DIR.exists() else []
        if not policy_files:
            st.info("No policy files found.")
        else:
            for pf in policy_files:
                policy = load_json(pf)
                with st.expander(pf.stem, expanded=True):
                    for key, section in policy.items():
                        if key == "version":
                            st.markdown(f"**Version:** {section}")
                            continue
                        if isinstance(section, dict):
                            actions = section.get("actions", [])
                            if actions:
                                st.markdown(f"**Category:** `{key}`")
                                action_rows = []
                                for a in actions:
                                    action_rows.append({
                                        "Action": a.get("name", "—"),
                                        "Eligibility": a.get("eligibility", "—"),
                                    })
                                st.dataframe(pd.DataFrame(action_rows), width='stretch', hide_index=True)

    with data_tab3:
        st.markdown("### Configuration Files")
        config_files = sorted(CONFIG_DIR.glob("*.json")) if CONFIG_DIR.exists() else []
        selected_config = st.selectbox("Select Config", [f.name for f in config_files], key="config_select")
        config_data = load_json(CONFIG_DIR / selected_config)

        if selected_config == "categories.json" and isinstance(config_data, list):
            st.markdown("**Complaint Categories & Keywords**")
            for entry in config_data:
                with st.expander(entry.get('name', '—'), expanded=True):
                    st.markdown(f"**Default Priority:** `{entry.get('default_priority', '—')}`")
                    st.markdown(f"**Keywords:** {', '.join(f'`{k}`' for k in entry.get('keywords', []))}")
        else:
            st.json(config_data)


# --------------- tab: Audit Trail ---------------

with tab_audit:
    st.subheader("Audit Trail")
    st.write("Compliance logs, audit entries, and SOX / GDPR validation results.")

    audit_dir = OUTPUTS_DIR / "audit_trail"
    if not audit_dir.exists():
        st.warning("No audit trail data found.")
    else:
        audit_files = sorted(audit_dir.glob("*.json"), reverse=True)
        if not audit_files:
            st.info("No audit files generated yet.")
        else:
            st.markdown(f"**Total Audit Reports:** {len(audit_files)}")

            selected_audit = st.selectbox("Select Audit Report", [f.stem for f in audit_files])
            audit_data = load_json(audit_dir / f"{selected_audit}.json")

            if audit_data:
                # Summary — handle both formats
                summary = audit_data.get("audit_summary", audit_data.get("summary", {}))
                if isinstance(summary, dict):
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Entries", summary.get("total_entries", audit_data.get("total_entries", "—")))
                    m2.metric("Valid", summary.get("valid_entries", audit_data.get("valid_entries", "—")))
                    m3.metric("Compliant", summary.get("compliant_entries", audit_data.get("compliant_entries", "—")))
                    m4.metric("Compliance Status", audit_data.get("compliance_status", "—"))

                st.markdown("---")

                # Compliance checks
                checks = audit_data.get("compliance_checks", {})
                if isinstance(checks, dict):
                    status = checks.get("status", "—")
                    passed = checks.get("checks_passed", "—")
                    total = checks.get("checks_total", "—")
                    icon = "✅" if status == "pass" else "❌"
                    st.markdown(f"**Compliance:** {icon} {status} ({passed}/{total} checks passed)")
                elif isinstance(checks, list):
                    for check in checks:
                        if isinstance(check, dict):
                            icon = "✅" if check.get("status") == "pass" else "❌"
                            st.markdown(f"{icon} **{check.get('check', '—')}** — {check.get('status', '—')}")

                # Audit trail entries
                entries = audit_data.get("audit_trail_entries", audit_data.get("audit_entries", []))
                if entries:
                    st.markdown("### Audit Trail Entries")
                    display_rows = []
                    for e in entries:
                        if isinstance(e, dict):
                            display_rows.append({
                                "Step": e.get("workflow_step", "—"),
                                "Agent": e.get("agent_id", "—"),
                                "Status": e.get("status", "—"),
                                "Validation": e.get("validation_status", "—"),
                                "Timestamp": e.get("timestamp", "—"),
                            })
                    if display_rows:
                        st.dataframe(pd.DataFrame(display_rows), width='stretch', hide_index=True)

                # Download CSV
                csv_path = audit_dir / f"{selected_audit}_index.csv"
                if csv_path.exists():
                    csv_content = csv_path.read_text(encoding="utf-8")
                    st.download_button("Download Audit CSV", csv_content, f"{selected_audit}.csv", "text/csv")

                # Exception log
                exception_file = PROJECT_ROOT / "logs" / "audit" / "exceptions.jsonl"
                if exception_file.exists():
                    st.markdown("---")
                    st.subheader("Exception Log")
                    lines = exception_file.read_text(encoding="utf-8").strip().split("\n")
                    if lines and lines[0]:
                        exceptions = []
                        for line in lines[-10:]:
                            try:
                                exceptions.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                        if exceptions:
                            st.dataframe(pd.DataFrame(exceptions), width='stretch', hide_index=True)
                        else:
                            st.success("No exceptions logged.")
                    else:
                        st.success("No exceptions logged.")

                st.markdown("---")
                with st.expander("Raw Audit JSON"):
                    st.json(audit_data)


st.markdown("---")
st.caption("Customer Service – Complaint & Issue Management  ·  Autogen 0.4  ·  Streamlit  ·  FastAPI")
