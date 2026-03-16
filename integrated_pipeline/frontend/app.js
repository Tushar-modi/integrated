/* ═══════════════════════════════════════════════
   Service Issue Resolution Agent — Frontend App
   All API calls via browser fetch() — no PowerShell.
   ═══════════════════════════════════════════════ */

const API = '';  // same origin
const $ = (sel) => document.querySelector(sel);

/* ── Agent endpoint map ──────────────────────── */

const AGENT_ENDPOINTS = {
  1: { path: '/api/agent1/classify', label: 'Classify' },
  2: { path: '/api/agent2/collect',  label: 'Evidence' },
  3: { path: '/api/agent3/plan',     label: 'Remedy' },
  4: { path: '/api/agent4/credit',   label: 'Credit' },
  5: { path: '/api/agent5/communicate', label: 'Communicate' },
  6: { path: '/api/agent6/audit',       label: 'Audit' },
};

/* ── State ───────────────────────────────────── */

let rawData = {};      // last result per agent
let pipelineResult = null;

/* ── Init ────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadSamples();
  $('#footer-ts').textContent = new Date().toLocaleString();
});

/* ── Health check ────────────────────────────── */

async function checkHealth() {
  const badge = $('#health-badge');
  try {
    const res = await fetch(API + '/health');
    if (res.ok) {
      badge.textContent = 'Healthy';
      badge.className = 'badge badge-ok';
    } else {
      badge.textContent = 'Unhealthy';
      badge.className = 'badge badge-err';
    }
  } catch {
    badge.textContent = 'Offline';
    badge.className = 'badge badge-err';
  }
}

/* ── Load available samples ──────────────────── */

async function loadSamples() {
  try {
    const res = await fetch(API + '/api/samples');
    if (res.ok) {
      const data = await res.json();
      const sel = $('#sample-select');
      sel.innerHTML = '';
      (data.samples || []).forEach(s => {
        const opt = document.createElement('option');
        opt.value = s;
        opt.textContent = s;
        sel.appendChild(opt);
      });
    }
  } catch { /* keep default */ }
  loadInputPreview();
}

/* ── Upload input files ──────────────────────── */

async function uploadInputs() {
  const complaint = $('#file-complaint').files[0];
  const profile = $('#file-profile').files[0];
  const history = $('#file-history').files[0];
  const statusEl = $('#upload-status');

  if (!complaint || !profile || !history) {
    statusEl.textContent = 'Please select all 3 files.';
    statusEl.className = 'upload-msg err';
    return;
  }

  statusEl.textContent = 'Uploading…';
  statusEl.className = 'upload-msg';
  $('#btn-upload').disabled = true;

  const formData = new FormData();
  formData.append('complaint_text', complaint);
  formData.append('customer_profile', profile);
  formData.append('interaction_history', history);

  try {
    const res = await fetch(API + '/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = data.detail || 'Upload failed';
      statusEl.className = 'upload-msg err';
      return;
    }
    // Add new sample to dropdown and select it
    const sel = $('#sample-select');
    const opt = document.createElement('option');
    opt.value = data.sample;
    opt.textContent = data.sample;
    sel.appendChild(opt);
    sel.value = data.sample;
    loadInputPreview();
    statusEl.textContent = `Uploaded as ${data.sample}`;
    statusEl.className = 'upload-msg ok';
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.className = 'upload-msg err';
  } finally {
    $('#btn-upload').disabled = false;
  }
}

/* ── Load input preview ──────────────────────── */

async function loadInputPreview() {
  const sample = $('#sample-select').value;
  try {
    const res = await fetch(API + `/api/inputs/${sample}`);
    if (res.ok) {
      const data = await res.json();
      $('#input-complaint').textContent = data.complaint_text || '(empty)';
      $('#input-profile').textContent = JSON.stringify(data.customer_profile, null, 2);
      $('#input-history').textContent = JSON.stringify(data.interaction_history, null, 2);
    } else {
      $('#input-complaint').textContent = '(could not load)';
    }
  } catch {
    $('#input-complaint').textContent = '(server offline)';
    $('#input-profile').textContent = '';
    $('#input-history').textContent = '';
  }
}

/* ── Run single agent ────────────────────────── */

async function runAgent(num) {
  const sample = $('#sample-select').value;
  const ep = AGENT_ENDPOINTS[num];
  const btn = document.querySelector(`.btn-agent[data-agent="${num}"]`);
  const statusEl = $(`#status-agent${num}`);
  const bodyEl = $(`#body-agent${num}`);

  setRunning(btn, statusEl, bodyEl);

  try {
    const res = await fetch(API + ep.path + `?sample=${sample}`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      setError(btn, statusEl, bodyEl, data.detail || 'Error');
      return;
    }
    rawData[num] = data;
    renderAgentResult(num, data);
    setDone(btn, statusEl);
  } catch (err) {
    setError(btn, statusEl, bodyEl, err.message);
  }
}

/* ── Run full pipeline ───────────────────────── */

async function runPipeline() {
  const sample = $('#sample-select').value;
  const pipeBtn = $('#btn-pipeline');
  pipeBtn.disabled = true;
  pipeBtn.textContent = '⏳ Running Pipeline…';

  for (let i = 1; i <= 6; i++) {
    const btn = document.querySelector(`.btn-agent[data-agent="${i}"]`);
    const statusEl = $(`#status-agent${i}`);
    const bodyEl = $(`#body-agent${i}`);
    setRunning(btn, statusEl, bodyEl);
  }

  try {
    const res = await fetch(API + `/api/pipeline/run?sample=${sample}`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      pipeBtn.textContent = '▶ Run Full Pipeline';
      pipeBtn.disabled = false;
      for (let i = 1; i <= 6; i++) {
        const btn = document.querySelector(`.btn-agent[data-agent="${i}"]`);
        const statusEl = $(`#status-agent${i}`);
        const bodyEl = $(`#body-agent${i}`);
        setError(btn, statusEl, bodyEl, data.detail || 'Pipeline error');
      }
      return;
    }
    pipelineResult = data;
    rawData = {};

    // Render pipeline summary
    renderPipelineSummary(data.pipeline_run);

    // Render each agent result
    const r = data.results || {};
    if (r.agent1_classifier) {
      rawData[1] = r.agent1_classifier;
      renderAgentResult(1, r.agent1_classifier);
      setDone(document.querySelector('.btn-agent[data-agent="1"]'), $('#status-agent1'));
    }
    if (r.agent2_evidence) {
      rawData[2] = r.agent2_evidence;
      renderAgentResult(2, r.agent2_evidence);
      setDone(document.querySelector('.btn-agent[data-agent="2"]'), $('#status-agent2'));
    }
    if (r.agent3_remedy) {
      rawData[3] = r.agent3_remedy;
      renderAgentResult(3, r.agent3_remedy);
      setDone(document.querySelector('.btn-agent[data-agent="3"]'), $('#status-agent3'));
    }
    if (r.agent4_credit) {
      rawData[4] = r.agent4_credit;
      renderAgentResult(4, r.agent4_credit);
      setDone(document.querySelector('.btn-agent[data-agent="4"]'), $('#status-agent4'));
    }
    if (r.agent5_communicator) {
      rawData[5] = r.agent5_communicator;
      renderAgentResult(5, r.agent5_communicator);
      setDone(document.querySelector('.btn-agent[data-agent="5"]'), $('#status-agent5'));
    }
    if (r.agent6_audit) {
      rawData[6] = r.agent6_audit;
      renderAgentResult(6, r.agent6_audit);
      setDone(document.querySelector('.btn-agent[data-agent="6"]'), $('#status-agent6'));
    }

    // Check for failed agents
    if (data.pipeline_run.failed_at) {
      const failIdx = { agent1_classifier: 1, agent2_evidence: 2, agent3_remedy: 3, agent4_credit: 4, agent5_communicator: 5, agent6_audit: 6 };
      const fi = failIdx[data.pipeline_run.failed_at];
      if (fi) {
        const errData = r[data.pipeline_run.failed_at] || {};
        setError(
          document.querySelector(`.btn-agent[data-agent="${fi}"]`),
          $(`#status-agent${fi}`),
          $(`#body-agent${fi}`),
          errData.error || 'Failed'
        );
      }
    }

    // Show raw JSON
    showRaw(data);

  } catch (err) {
    for (let i = 1; i <= 6; i++) {
      setError(
        document.querySelector(`.btn-agent[data-agent="${i}"]`),
        $(`#status-agent${i}`),
        $(`#body-agent${i}`),
        err.message
      );
    }
  } finally {
    pipeBtn.textContent = '▶ Run Full Pipeline';
    pipeBtn.disabled = false;
  }
}

/* ═══ RENDERERS ═══════════════════════════════ */

function renderAgentResult(num, data) {
  const bodyEl = $(`#body-agent${num}`);
  switch (num) {
    case 1: renderAgent1(bodyEl, data); break;
    case 2: renderAgent2(bodyEl, data); break;
    case 3: renderAgent3(bodyEl, data); break;
    case 4: renderAgent4(bodyEl, data); break;
    case 5: renderAgent5(bodyEl, data); break;
    case 6: renderAgent6(bodyEl, data); break;
  }
}

function renderAgent1(el, d) {
  const c = d.complaint_category || d;
  el.innerHTML = `
    ${row('Complaint ID', c.complaint_id)}
    ${row('Category', c.category)}
    ${row('Severity', c.severity_level)}
    ${row('Priority', c.priority_tag)}
    ${row('Routing Queue', c.routing_queue)}
    ${row('Accuracy', (c.accuracy_score != null ? (c.accuracy_score * 100).toFixed(0) + '%' : '—'))}
    ${row('Validation', c.validation_status, c.validation_status)}
    ${row('Mode', d.complaint_category ? '—' : (d.mode || '—'))}
  `;
}

function renderAgent2(el, d) {
  const order = d.sales_order;
  const delivery = d.delivery;
  el.innerHTML = `
    ${row('Complaint ID', d.complaint_id)}
    ${row('Customer ID', d.customer_id)}
    ${row('Sales Order', order ? (order.VBELN || order.VBAK?.VBELN || 'found') : 'none')}
    ${row('Delivery', delivery ? (delivery.VBELN || delivery.LIKP?.VBELN || 'found') : 'none')}
    ${row('Billing', d.billing ? 'found' : 'none')}
    ${row('Completeness', d.completeness_flag ? 'Complete' : 'Incomplete')}
    ${row('Validation', d.validation_status, d.validation_status)}
    ${row('Mode', d.mode || '—')}
  `;
}

function renderAgent3(el, d) {
  const actions = (d.actions || []).map(a =>
    `<span style="color:var(--text)">${a.type}</span>`
  ).join(', ') || 'none';
  el.innerHTML = `
    ${row('Complaint ID', d.complaint_id)}
    ${row('Category', d.category)}
    ${row('Actions', actions)}
    ${row('Cost', d.cost_estimate ? `${d.cost_estimate.amount} ${d.cost_estimate.currency}` : '—')}
    ${row('Policy Compliance', d.policy_compliance ? 'Yes' : 'No')}
    ${row('Validation', d.validation_status, d.validation_status)}
    ${row('Mode', d.mode || '—')}
  `;
}

function renderAgent4(el, d) {
  const a = d.approval || {};
  const sapDoc = d.sap_fi_posting?.BKPF?.BELNR || '—';
  const statusClass = a.status === 'approved' ? 'pass' : a.status === 'not_applicable' ? 'pass' : 'fail';
  el.innerHTML = `
    ${row('Complaint ID', d.complaint_id)}
    ${row('Credit Status', a.status, statusClass)}
    ${row('Amount', a.amount != null ? `${a.amount} ${a.currency}` : '—')}
    ${a.credit_types ? row('Credit Types', a.credit_types.join(', ')) : ''}
    ${row('Credit Doc', a.credit_doc || '—')}
    ${a.reason ? row('Reason', a.reason) : ''}
    ${row('SAP FI Doc', sapDoc)}
    ${row('Conditions', (a.conditions || []).join(', ') || '—')}
    ${row('Validation', d.validation_status, d.validation_status)}
    ${row('Mode', d.mode || '—')}
  `;
}

function renderAgent5(el, d) {
  const to = d.to || {};
  const comp = d.compliance || {};
  el.innerHTML = `
    ${row('Complaint ID', d.complaint_id)}
    ${row('Recipient', to.name || '—')}
    ${row('Email', to.email || '—')}
    ${row('Channel', d.dispatch_channel)}
    ${row('GDPR', comp.gdpr ? 'Pass' : 'Fail', comp.gdpr ? 'pass' : 'fail')}
    ${row('Brand', comp.brand ? 'Pass' : 'Fail', comp.brand ? 'pass' : 'fail')}
    ${row('Validation', d.validation_status, d.validation_status)}
    ${row('Mode', d.mode || '—')}
    <div style="margin-top:0.5rem">
      <div class="detail-label" style="margin-bottom:0.25rem">Message Body</div>
      <div class="msg-body">${escapeHtml(d.body || '—')}</div>
    </div>
  `;
}

function renderAgent6(el, d) {
  const s = d.audit_summary || {};
  const c = d.compliance || {};
  const exc = d.exceptions || [];
  const trail = d.audit_trail_entries || [];
  const missing = (s.missing_steps || []).join(', ') || 'none';

  // Summary rows
  let html = `
    ${row('Complaint ID', d.complaint_id)}
    ${row('Customer ID', d.customer_id)}
    ${row('Total Entries', s.total_entries)}
    ${row('Valid / Compliant', `${s.valid_entries} / ${s.compliant_entries}`)}
    ${row('Validation Rate', s.validation_rate != null ? (s.validation_rate * 100).toFixed(0) + '%' : '—')}
    ${row('Compliance Rate', s.compliance_rate != null ? (s.compliance_rate * 100).toFixed(0) + '%' : '—')}
    ${row('All Steps Done', s.all_steps_completed ? 'Yes' : 'No', s.all_steps_completed ? 'pass' : 'fail')}
    ${row('Missing Steps', missing)}
    ${row('Compliance', c.status, c.status === 'pass' ? 'pass' : 'fail')}
    ${row('Checks Passed', c.checks_passed != null ? `${c.checks_passed} / ${c.checks_total}` : '—')}
    ${row('Exceptions', exc.length)}
    ${row('Mode', d.mode || '—')}
  `;

  // Audit trail table
  if (trail.length > 0) {
    html += `
      <div style="margin-top:0.75rem; padding-top:0.5rem; border-top:1px solid var(--border)">
        <div class="detail-label" style="margin-bottom:0.25rem; text-transform:uppercase; letter-spacing:0.03em; font-size:0.72rem">Audit Trail</div>
        <table class="audit-table">
          <thead><tr>
            <th>Step</th><th>Agent</th><th>Status</th><th>Validation</th><th>Flags</th>
          </tr></thead>
          <tbody>
    `;
    for (const e of trail) {
      const vCls = e.validation_status === 'pass' ? 'v-pass' : 'v-fail';
      const flags = (e.compliance_flags || []).join(', ') || '—';
      html += `<tr>
        <td>${e.workflow_step}</td>
        <td>${e.agent_id}</td>
        <td>${e.status}</td>
        <td class="${vCls}">${e.validation_status}</td>
        <td>${flags}</td>
      </tr>`;
    }
    html += `</tbody></table>`;

    // CSV download link
    if (d.complaint_id) {
      html += `<a class="btn-csv" href="/api/audit-csv/${encodeURIComponent(d.complaint_id)}" download>Download Audit CSV</a>`;
    }
    html += `</div>`;
  }

  el.innerHTML = html;
}

function renderPipelineSummary(run) {
  const section = $('#pipeline-summary');
  section.classList.remove('hidden');
  const grid = $('#summary-grid');
  grid.innerHTML = `
    ${summaryItem('Status', run.status === 'completed' ? '✓ Completed' : '✗ Failed', run.status === 'completed' ? 'ok' : 'err')}
    ${summaryItem('Complaint', run.complaint_id || '—', '')}
    ${summaryItem('Agents Run', run.agents_executed ? run.agents_executed.length + ' / 6' : '—', '')}
    ${summaryItem('Elapsed', run.elapsed_seconds + 's', '')}
    ${run.failed_at ? summaryItem('Failed At', run.failed_at, 'err') : ''}
  `;
}

/* ═══ HELPERS ═════════════════════════════════ */

function row(label, value, cls) {
  const valClass = cls ? ` ${cls}` : '';
  return `<div class="detail-row">
    <span class="detail-label">${label}</span>
    <span class="detail-value${valClass}">${value ?? '—'}</span>
  </div>`;
}

function summaryItem(label, value, cls) {
  return `<div class="summary-item">
    <div class="label">${label}</div>
    <div class="value ${cls}">${value}</div>
  </div>`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function setRunning(btn, statusEl, bodyEl) {
  if (btn) btn.classList.add('running');
  if (statusEl) { statusEl.textContent = 'Running…'; statusEl.className = 'card-status running'; }
  if (bodyEl) bodyEl.innerHTML = '<p class="placeholder">Running…</p>';
}

function setDone(btn, statusEl) {
  if (btn) { btn.classList.remove('running'); btn.disabled = false; }
  if (statusEl) { statusEl.textContent = 'Pass'; statusEl.className = 'card-status pass'; }
}

function setError(btn, statusEl, bodyEl, msg) {
  if (btn) { btn.classList.remove('running'); btn.disabled = false; }
  if (statusEl) { statusEl.textContent = 'Error'; statusEl.className = 'card-status fail'; }
  if (bodyEl) bodyEl.innerHTML = `<p style="color:var(--red)">${escapeHtml(msg)}</p>`;
}

function showRaw(data) {
  const sec = $('#raw-section');
  sec.classList.remove('hidden');
  $('#raw-json').textContent = JSON.stringify(data, null, 2);
}

function toggleRaw() {
  const pre = $('#raw-json');
  pre.classList.toggle('hidden');
}
