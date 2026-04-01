/**
 * DataPrep Engine — app.js
 * Full integration with Flask app.py backend
 * Handles: upload → profile → recommend → preprocess → export & report
 */

'use strict';

// ═══════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════
const State = {
  filename:        null,
  profile:         null,
  recommendations: null,
  processResult:   null,
  sessionStart:    Date.now(),
  apiKey:          null,
  provider:        'none',
};

const API_BASE = 'http://127.0.0.1:5000';

// ═══════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  startSessionClock();
  updateProviderLabel();
});

// ═══════════════════════════════════════════
//  SESSION CLOCK
// ═══════════════════════════════════════════
function startSessionClock() {
  const el = document.getElementById('sessionClock');
  setInterval(() => {
    const diff = Math.floor((Date.now() - State.sessionStart) / 1000);
    const h = String(Math.floor(diff / 3600)).padStart(2, '0');
    const m = String(Math.floor((diff % 3600) / 60)).padStart(2, '0');
    const s = String(diff % 60).padStart(2, '0');
    el.textContent = `${h}:${m}:${s}`;
  }, 1000);
}

// ═══════════════════════════════════════════
//  TABS
// ═══════════════════════════════════════════
function switchTab(tabId, btn) {
  if (btn && btn.classList.contains('locked')) return;
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.wf-tab').forEach(b => b.classList.remove('active'));
  document.getElementById(tabId)?.classList.add('active');
  if (btn) btn.classList.add('active');
  updatePipelineMini(tabId);
}

function unlockTab(tabId) {
  const btn = document.querySelector(`[data-tab="${tabId}"]`);
  if (btn) {
    btn.classList.remove('locked');
    btn.classList.add('done');
  }
}

function updatePipelineMini(tabId) {
  const map = {
    'tab-upload':    'pip-upload',
    'tab-profile':   'pip-profile',
    'tab-recommend': 'pip-recommend',
    'tab-process':   'pip-process',
    'tab-export':    'pip-export',
  };
  const order = ['tab-upload','tab-profile','tab-recommend','tab-process','tab-export'];
  const activeIdx = order.indexOf(tabId);
  order.forEach((t, i) => {
    const node = document.getElementById(map[t]);
    if (!node) return;
    node.className = 'pip-node';
    if (i < activeIdx) node.classList.add('done');
    if (i === activeIdx) node.classList.add('active');
  });
}

// ═══════════════════════════════════════════
//  PROVIDER TOGGLE
// ═══════════════════════════════════════════
function updateProviderLabel() {
  const sel = document.getElementById('providerSelect');
  const row = document.getElementById('apiKeyRow');
  const note = document.getElementById('apiNote');
  State.provider = sel.value;
  document.getElementById('s-provider').textContent =
    sel.value === 'anthropic' ? 'Anthropic Claude' :
    sel.value === 'groq'      ? 'Groq / Mixtral' : 'Rule-Based';

  if (sel.value === 'none') {
    row.style.display = 'none';
    note.textContent = 'Using intelligent rule-based recommendations — no API key needed.';
  } else {
    row.style.display = 'flex';
    note.textContent = sel.value === 'anthropic'
      ? 'Claude will analyze your dataset and generate contextual preprocessing strategies.'
      : 'Groq (Mixtral-8x7b) will analyze your dataset — fast & free tier available.';
  }
}

// ═══════════════════════════════════════════
//  FILE UPLOAD
// ═══════════════════════════════════════════
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.add('dragging');
}
function handleDragLeave(e) {
  document.getElementById('uploadZone').classList.remove('dragging');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.remove('dragging');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
}
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) uploadFile(file);
}

function uploadFile(file) {
  const allowed = ['csv','xlsx','xls'];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showToast('Only CSV, XLSX, XLS files are supported.', 'error');
    return;
  }

  State.apiKey  = document.getElementById('apiKeyInput').value.trim() || null;
  State.provider = document.getElementById('providerSelect').value;

  showProgress(true);
  animateProgressBar(0, 40, 600);

  const formData = new FormData();
  formData.append('file', file);

  fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      animateProgressBar(40, 100, 400);
      setTimeout(() => {
        showProgress(false);
        if (data.error) { showToast(data.error, 'error'); return; }
        State.filename = data.profile.filename;
        State.profile  = data.profile;
        renderUploadSuccess(file, data.profile);
        updateSidebarSession(data.profile);
        showToast('Dataset uploaded and analyzed!', 'success');
      }, 500);
    })
    .catch(err => {
      showProgress(false);
      showToast('Upload failed. Is the Flask server running on port 5000?', 'error');
      console.error(err);
    });
}

function showProgress(show) {
  document.getElementById('uploadProgress').style.display = show ? 'block' : 'none';
  document.getElementById('uploadedInfo').style.display = 'none';
}

function animateProgressBar(from, to, duration) {
  const bar = document.getElementById('upBar');
  const status = document.getElementById('upStatus');
  const start = performance.now();
  function frame(now) {
    const t = Math.min((now - start) / duration, 1);
    const val = from + (to - from) * t;
    bar.style.width = val + '%';
    status.textContent = val < 50 ? 'Uploading...' : val < 90 ? 'Profiling dataset...' : 'Almost done...';
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function renderUploadSuccess(file, profile) {
  const info = document.getElementById('uploadedInfo');
  info.style.display = 'block';
  document.getElementById('ui-filename').textContent = file.name;
  document.getElementById('ui-meta').textContent =
    `${profile.basic_info.rows.toLocaleString()} rows × ${profile.basic_info.columns} columns · ${profile.basic_info.memory_usage} · ${file.type || 'text/csv'}`;
}

// ═══════════════════════════════════════════
//  SIDEBAR
// ═══════════════════════════════════════════
function updateSidebarSession(profile) {
  document.getElementById('s-status').textContent   = 'Ready';
  document.getElementById('s-status').className     = 'ir-val green';
  document.getElementById('s-filename').textContent = profile.filename;
  document.getElementById('s-rows').textContent     = profile.basic_info.rows.toLocaleString();
  document.getElementById('s-cols').textContent     = profile.basic_info.columns;
  document.getElementById('s-memory').textContent   = profile.basic_info.memory_usage;
  document.getElementById('s-dupes').textContent    = profile.basic_info.duplicate_rows;

  renderQualityWidget(profile.quality_score);
  renderIssuesWidget(profile.potential_issues);

  document.getElementById('qa-recommend').style.display = 'block';
}

function renderQualityWidget(qs) {
  document.getElementById('quality-widget').style.display = 'block';
  const score = qs.overall;
  const circle = document.getElementById('qr-circle');
  const circumference = 264;
  circle.style.strokeDashoffset = circumference - (circumference * score / 100);
  const colorMap = { Excellent: '#1a7a3c', Good: '#1a7a3c', Fair: '#b87a10', Poor: '#c0160f' };
  circle.style.stroke = colorMap[qs.rating] || '#e8620a';
  document.getElementById('qr-score').textContent = Math.round(score);
  document.getElementById('qr-rating').textContent = qs.rating;

  setBar('qb-completeness', 'qbv-completeness', qs.completeness);
  setBar('qb-consistency',  'qbv-consistency',  qs.consistency);
  setBar('qb-validity',     'qbv-validity',     qs.validity);
}

function setBar(barId, valId, val) {
  document.getElementById(barId).style.width = Math.min(val, 100) + '%';
  document.getElementById(valId).textContent = Math.round(val) + '%';
}

function renderIssuesWidget(issues) {
  if (!issues || issues.length === 0) return;
  document.getElementById('issues-widget').style.display = 'block';
  const list = document.getElementById('issues-list');
  list.innerHTML = issues.slice(0, 6).map(iss => `
    <div class="issue-item ${iss.severity}">
      <div class="issue-sev">${iss.severity.toUpperCase()}</div>
      <div>${iss.description}</div>
    </div>`).join('');
}

// ═══════════════════════════════════════════
//  PROFILE TAB
// ═══════════════════════════════════════════
function proceedToProfile() {
  if (!State.profile) return;
  renderProfileTab(State.profile);
  unlockTab('tab-profile');
  switchTab('tab-profile', document.getElementById('btn-tab-profile'));
}

function renderProfileTab(profile) {
  renderOverviewCards(profile);
  renderColumnTable(profile.column_analysis);
  renderMissingChart(profile.missing_values);
  renderStatsTable(profile.statistical_summary);
  renderMLSection(profile.ml_problem_type);
}

function renderOverviewCards(profile) {
  const qs = profile.quality_score;
  const cards = [
    { icon:'📊', val: profile.basic_info.rows.toLocaleString(), label:'Total Rows', cls:'' },
    { icon:'📋', val: profile.basic_info.columns, label:'Columns', cls:'' },
    { icon:'🔴', val: profile.missing_values.missing_percentage.toFixed(1)+'%', label:'Missing Data', cls: profile.missing_values.missing_percentage > 10 ? 'red' : 'green' },
    { icon:'📝', val: profile.basic_info.duplicate_rows, label:'Duplicate Rows', cls: profile.basic_info.duplicate_rows > 0 ? 'red' : 'green' },
    { icon:'⚡', val: Math.round(qs.overall), label:'Quality Score', cls: qs.overall >= 60 ? 'green' : qs.overall >= 40 ? 'orange' : 'red' },
    { icon:'💾', val: profile.basic_info.memory_usage, label:'Memory Usage', cls:'' },
    { icon:'🔢', val: profile.data_types.numeric, label:'Numeric Cols', cls:'ask' },
    { icon:'🔤', val: profile.data_types.categorical, label:'Categorical Cols', cls:'orange' },
  ];
  document.getElementById('overviewGrid').innerHTML = cards.map(c => `
    <div class="ov-card">
      <div class="ov-card-icon">${c.icon}</div>
      <div class="ov-card-value ${c.cls}">${c.val}</div>
      <div class="ov-card-label">${c.label}</div>
    </div>`).join('');
}

function renderColumnTable(cols) {
  document.getElementById('columnTableBody').innerHTML = cols.map(col => {
    const missingClass = col.missing_percentage > 30 ? 'red' : col.missing_percentage > 5 ? 'orange' : 'green';
    const typeLabel = col.dtype.includes('int') || col.dtype.includes('float') ? 'numeric' :
                      col.dtype.includes('object') ? 'object' : 'datetime';
    const flag = col.is_constant ? '<span class="flag-badge constant">CONST</span>' :
                 col.is_identifier ? '<span class="flag-badge ident">ID</span>' :
                 '<span class="flag-badge ok">OK</span>';
    return `<tr>
      <td class="mono">${col.name}</td>
      <td><span class="type-badge ${typeLabel}">${col.dtype}</span></td>
      <td class="${missingClass} mono">${col.missing_count}</td>
      <td class="${missingClass} mono">${col.missing_percentage.toFixed(1)}%</td>
      <td class="mono">${col.unique_values}</td>
      <td>${flag}</td>
      <td class="sample-vals">${col.sample_values.slice(0, 3).join(', ')}</td>
    </tr>`;
  }).join('');
}

function renderMissingChart(mv) {
  if (!mv.columns_with_missing || mv.columns_with_missing.length === 0) return;
  document.getElementById('missingSection').style.display = 'block';
  document.getElementById('missingChart').innerHTML = mv.columns_with_missing.map(col => {
    const pct = col.percentage;
    const cls = pct > 30 ? 'high' : pct > 10 ? 'medium' : 'low';
    return `<div class="missing-bar-item">
      <div class="mb-col" title="${col.column}">${col.column}</div>
      <div class="mb-track"><div class="mb-fill ${cls}" style="width:${Math.min(pct,100)}%"></div></div>
      <div class="mb-pct">${pct.toFixed(1)}%</div>
    </div>`;
  }).join('');
}

function renderStatsTable(stats) {
  if (!stats || Object.keys(stats).length === 0) return;
  document.getElementById('statsSection').style.display = 'block';
  const cols = Object.keys(stats);
  const metrics = ['count','mean','std','min','25%','50%','75%','max'];
  document.getElementById('statsHead').innerHTML =
    '<th>Metric</th>' + cols.map(c => `<th>${c}</th>`).join('');
  document.getElementById('statsBody').innerHTML = metrics.map(m => `
    <tr>
      <td class="mono" style="font-weight:600;color:var(--text-mid)">${m}</td>
      ${cols.map(c => `<td class="mono">${stats[c][m] !== undefined ? Number(stats[c][m]).toFixed(3) : '—'}</td>`).join('')}
    </tr>`).join('');
}

function renderMLSection(ml) {
  if (!ml || !ml.potential_targets || ml.potential_targets.length === 0) return;
  document.getElementById('mlSection').style.display = 'block';
  document.getElementById('mlTargets').innerHTML = `
    <div class="ml-target-cards">
      ${ml.potential_targets.slice(0, 6).map(t => `
        <div class="ml-card">
          <div class="ml-card-col">📌 ${t.column}</div>
          <span class="ml-card-type ${t.type}">${t.type.replace(/_/g,' ')}</span>
          <div class="ml-card-conf">Confidence: <strong>${t.confidence}</strong> · ${t.unique_values} unique values</div>
        </div>`).join('')}
    </div>`;
}

// ═══════════════════════════════════════════
//  RECOMMENDATIONS TAB
// ═══════════════════════════════════════════
function proceedToRecommend() {
  unlockTab('tab-recommend');
  switchTab('tab-recommend', document.getElementById('btn-tab-recommend'));
  if (State.recommendations) {
    renderRecommendations(State.recommendations);
    return;
  }
  fetchRecommendations();
}

function fetchRecommendations() {
  document.getElementById('recLoading').style.display = 'flex';
  document.getElementById('recContent').style.display = 'none';
  document.getElementById('recFooter').style.display  = 'none';

  const body = {
    profile:  State.profile,
    api_key:  State.apiKey,
    provider: State.provider === 'none' ? 'anthropic' : State.provider,
  };

  fetch(`${API_BASE}/api/recommend`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
    .then(r => r.json())
    .then(data => {
      document.getElementById('recLoading').style.display = 'none';
      if (data.error) { showToast(data.error, 'error'); return; }
      State.recommendations = data.recommendations;
      renderRecommendations(data.recommendations);
      document.getElementById('qa-process').style.display = 'block';
      document.getElementById('qa-report').style.display = 'block';
      showToast('Recommendations generated!', 'success');
    })
    .catch(() => {
      document.getElementById('recLoading').style.display = 'none';
      showToast('Failed to get recommendations. Check server connection.', 'error');
    });
}

function renderRecommendations(rec) {
  document.getElementById('recContent').style.display = 'block';
  document.getElementById('recFooter').style.display  = 'block';

  const src = rec.source || 'rule_based';
  document.getElementById('recSourceTag').textContent =
    src.includes('anthropic') ? '✦ Claude AI' :
    src.includes('groq')      ? '⚡ Groq AI' : '⚙ Rule-Based';

  // Assessment
  document.getElementById('assessmentBanner').textContent =
    rec.overall_assessment || 'No assessment provided.';

  // Suggestions
  const sugList = rec.suggestions || [];
  document.getElementById('suggestionsList').innerHTML = sugList.length
    ? sugList.map(s => `
      <div class="suggestion-item ${s.severity}">
        <div class="sug-icon">${s.severity==='high' ? '🔴' : s.severity==='medium' ? '🟡' : '🔵'}</div>
        <div class="sug-body">
          <div class="sug-sev ${s.severity}">${s.severity.toUpperCase()}</div>
          <div class="sug-text">${s.text}</div>
          ${s.impact ? `<div class="sug-impact">Impact: ${s.impact}</div>` : ''}
        </div>
      </div>`).join('')
    : '<div style="color:var(--text-light);font-size:13px;font-style:italic">No major issues detected.</div>';

  // Columns to drop
  const drops = rec.columns_to_drop || [];
  document.getElementById('colsDropList').innerHTML = drops.length
    ? drops.map(d => `<div class="sc-item"><span class="sc-col">${d.column}</span><span class="sc-val red">${d.reason}</span></div>`).join('')
    : '<div class="sc-empty">No columns to drop.</div>';

  // Missing value strategy
  const mvs = rec.missing_value_strategy || {};
  document.getElementById('missingStratList').innerHTML = Object.entries(mvs).length
    ? Object.entries(mvs).map(([col, strat]) => `<div class="sc-item"><span class="sc-col">${col}</span><span class="sc-val orange">${strat}</span></div>`).join('')
    : '<div class="sc-empty">No missing value handling needed.</div>';

  // Encoding
  const enc = rec.encoding_recommendations || {};
  document.getElementById('encodingList').innerHTML = Object.entries(enc).length
    ? Object.entries(enc).map(([col, type]) => `<div class="sc-item"><span class="sc-col">${col}</span><span class="sc-val ask">${type.replace(/_/g,' ')}</span></div>`).join('')
    : '<div class="sc-empty">No encoding needed.</div>';

  // Scaling + split
  const split = rec.data_split || { train: 0.7, validation: 0.15, test: 0.15 };
  document.getElementById('scalingSplitInfo').innerHTML = `
    <div class="sc-item"><span class="sc-col">Scaler</span><span class="sc-val green">${rec.scaling_recommendation || 'standard'}</span></div>
    <div class="sc-item"><span class="sc-col">Outliers</span><span class="sc-val green">${rec.outlier_handling || 'iqr'}</span></div>
    <div class="sc-item"><span class="sc-col">Train</span><span class="sc-val orange">${(split.train * 100).toFixed(0)}%</span></div>
    <div class="sc-item"><span class="sc-col">Validation</span><span class="sc-val ask">${(split.validation * 100).toFixed(0)}%</span></div>
    <div class="sc-item"><span class="sc-col">Test</span><span class="sc-val green">${(split.test * 100).toFixed(0)}%</span></div>`;

  // Pipeline steps
  const steps = rec.preprocessing_steps || [];
  document.getElementById('pipelineSteps').innerHTML = steps.map((s, i) => `
    <div class="ps-item">
      <div class="ps-num">${String(i + 1).padStart(2, '0')}</div>
      <div>${s}</div>
    </div>`).join('');
}

// ═══════════════════════════════════════════
//  PREPROCESS TAB
// ═══════════════════════════════════════════
function proceedToProcess() {
  unlockTab('tab-process');
  switchTab('tab-process', document.getElementById('btn-tab-process'));

  document.getElementById('preprocessConfirm').style.display = 'block';
  document.getElementById('processLoading').style.display    = 'none';
  document.getElementById('processResult').style.display     = 'none';

  document.getElementById('pc-filename').textContent = State.filename || '';
  const steps = (State.recommendations?.preprocessing_steps || []).slice(0, 8);
  document.getElementById('pc-steps-list').innerHTML = steps.map(s => `
    <div class="pc-step-item">${s}</div>`).join('');
}

function runPreprocessing() {
  document.getElementById('preprocessConfirm').style.display = 'none';
  document.getElementById('processLoading').style.display    = 'flex';

  const logEl = document.getElementById('processLog');
  logEl.innerHTML = '';
  const steps = ['Initializing pipeline...','Dropping irrelevant columns...','Removing duplicates...','Handling missing values...','Encoding categoricals...','Capping outliers (IQR)...','Scaling features...','Splitting data...','Saving files...'];
  let si = 0;
  const logInterval = setInterval(() => {
    if (si < steps.length) {
      document.getElementById('processStep').textContent = steps[si];
      logEl.innerHTML += `<div>${steps[si]}</div>`;
      logEl.scrollTop = logEl.scrollHeight;
      si++;
    }
  }, 600);

  fetch(`${API_BASE}/api/preprocess`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename:        State.filename,
      recommendations: State.recommendations,
    }),
  })
    .then(r => r.json())
    .then(data => {
      clearInterval(logInterval);
      document.getElementById('processLoading').style.display = 'none';
      if (data.error) {
        document.getElementById('preprocessConfirm').style.display = 'block';
        showToast(data.error, 'error'); return;
      }
      State.processResult = data;
      document.getElementById('processResult').style.display = 'block';
      renderProcessResult(data);
      unlockTab('tab-export');
      showToast('Preprocessing complete!', 'success');
    })
    .catch(() => {
      clearInterval(logInterval);
      document.getElementById('processLoading').style.display = 'none';
      document.getElementById('preprocessConfirm').style.display = 'block';
      showToast('Preprocessing failed. Check server connection.', 'error');
    });
}

function renderProcessResult(data) {
  const shapes = data.shapes || {};
  document.getElementById('shapeCards').innerHTML = `
    <div class="shape-card train">
      <div class="shape-card-label">Train</div>
      <div class="shape-card-num">${shapes.train?.[0] ?? '—'}</div>
      <div class="shape-card-sub">${shapes.train?.[1] ?? ''} features</div>
    </div>
    <div class="shape-card val">
      <div class="shape-card-label">Validation</div>
      <div class="shape-card-num">${shapes.val?.[0] ?? '—'}</div>
      <div class="shape-card-sub">${shapes.val?.[1] ?? ''} features</div>
    </div>
    <div class="shape-card test">
      <div class="shape-card-label">Test</div>
      <div class="shape-card-num">${shapes.test?.[0] ?? '—'}</div>
      <div class="shape-card-sub">${shapes.test?.[1] ?? ''} features</div>
    </div>`;

  const logList = data.log || [];
  document.getElementById('logTimeline').innerHTML = logList.map((entry, i) => `
    <div class="lt-item">
      <div class="lt-num">#${String(i + 1).padStart(2, '0')}</div>
      <div class="lt-text">${entry}</div>
    </div>`).join('');

  // Update export tab metas
  if (shapes.train) document.getElementById('ec-train-meta').textContent = `${shapes.train[0]} rows × ${shapes.train[1]} cols`;
  if (shapes.val)   document.getElementById('ec-val-meta').textContent   = `${shapes.val[0]} rows × ${shapes.val[1]} cols`;
  if (shapes.test)  document.getElementById('ec-test-meta').textContent  = `${shapes.test[0]} rows × ${shapes.test[1]} cols`;

  ['dlTrainBtn','dlValBtn','dlTestBtn','dlAllBtn','genReportBtn','dlReportBtn'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = false;
  });
}

// ═══════════════════════════════════════════
//  EXPORT TAB
// ═══════════════════════════════════════════
function proceedToExport() {
  switchTab('tab-export', document.querySelector('[data-tab="tab-export"]'));
}

function downloadFile(split) {
  if (!State.processResult) return;
  const fileMap = { train: State.processResult.train_file, val: State.processResult.val_file, test: State.processResult.test_file };
  const filename = fileMap[split];
  if (!filename) return;
  triggerDownload(`${API_BASE}/api/download/${filename}`, filename);
  showToast(`Downloading ${split} dataset...`, 'info');
}

function downloadAll() {
  ['train','val','test'].forEach((s, i) => setTimeout(() => downloadFile(s), i * 600));
  showToast('Downloading all splits...', 'info');
}

function triggerDownload(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
}

// ═══════════════════════════════════════════
//  REPORT GENERATOR
// ═══════════════════════════════════════════
function generateReport() {
  if (!State.profile || !State.recommendations) {
    showToast('Complete preprocessing first.', 'warning');
    return;
  }
  document.getElementById('reportPreview').innerHTML = buildReportPreview();
  document.getElementById('dlReportBtn').disabled = false;
  showToast('Report generated!', 'success');
}

function buildReportPreview() {
  const p  = State.profile;
  const r  = State.recommendations;
  const pr = State.processResult;
  const qs = p.quality_score;
  const now = new Date().toLocaleString();
  return `
    <div class="report-summary">
      <strong>DataPrep Engine — Full Analysis Report</strong><br/>
      Generated: ${now}<br/><br/>
      <strong>Dataset:</strong> ${p.filename} &nbsp;|&nbsp;
      <strong>Rows:</strong> ${p.basic_info.rows.toLocaleString()} &nbsp;|&nbsp;
      <strong>Columns:</strong> ${p.basic_info.columns}<br/>
      <strong>Quality Score:</strong> <span style="color:var(--orange)">${qs.overall}/100 (${qs.rating})</span><br/>
      &emsp;• Completeness: ${qs.completeness}%<br/>
      &emsp;• Consistency: ${qs.consistency}%<br/>
      &emsp;• Validity: ${qs.validity}%<br/><br/>
      <strong>Missing Data:</strong> ${p.missing_values.missing_percentage.toFixed(2)}%<br/>
      <strong>Duplicates:</strong> ${p.basic_info.duplicate_rows}<br/>
      <strong>ML Task:</strong> ${p.ml_problem_type?.suggested_type || '—'}<br/><br/>
      <strong>Preprocessing Applied:</strong><br/>
      ${(r.preprocessing_steps || []).map((s, i) => `&emsp;${i + 1}. ${s}`).join('<br/>')}<br/><br/>
      ${pr ? `<strong>Output Shapes:</strong><br/>
      &emsp;Train: ${pr.shapes.train?.[0]} × ${pr.shapes.train?.[1]}<br/>
      &emsp;Val: ${pr.shapes.val?.[0]} × ${pr.shapes.val?.[1]}<br/>
      &emsp;Test: ${pr.shapes.test?.[0]} × ${pr.shapes.test?.[1]}<br/>` : ''}
      <br/><strong>Assessment:</strong> ${r.overall_assessment}
    </div>`;
}

function downloadReport() {
  if (!State.profile) return;
  const p  = State.profile;
  const r  = State.recommendations;
  const pr = State.processResult;
  const qs = p.quality_score;
  const now = new Date().toLocaleString();

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>DataPrep Report — ${p.filename}</title>
  <style>
    body{font-family:'DM Sans',sans-serif;background:#faf6f0;color:#2c2010;padding:48px;max-width:900px;margin:0 auto}
    h1{font-family:'Playfair Display',serif;color:#e8620a;font-size:36px;margin-bottom:4px}
    h2{font-family:'Playfair Display',serif;color:#1a1208;font-size:22px;margin:32px 0 12px;border-bottom:2px solid #e8dece;padding-bottom:8px}
    h3{color:#e8620a;font-size:14px;text-transform:uppercase;letter-spacing:1px;margin:18px 0 8px}
    .meta{color:#8a7862;font-size:13px;margin-bottom:32px}
    .score-box{display:flex;gap:32px;background:#fff;border:1px solid #d8cdb8;border-radius:10px;padding:24px;margin-bottom:24px}
    .score-main{text-align:center}
    .score-num{font-family:'Playfair Display',serif;font-size:64px;color:#e8620a;line-height:1}
    .score-label{color:#8a7862;font-size:12px;text-transform:uppercase;letter-spacing:1px}
    .score-bars{flex:1}
    .bar-row{display:flex;align-items:center;gap:12px;margin-bottom:10px}
    .bar-lbl{width:100px;font-size:12px;color:#5a4535}
    .bar-track{flex:1;height:8px;background:#e8dece;border-radius:4px;overflow:hidden}
    .bar-fill{height:100%;background:linear-gradient(90deg,#f07a2a,#e8620a);border-radius:4px}
    .bar-val{font-size:12px;color:#8a7862;width:36px;text-align:right;font-family:monospace}
    table{border-collapse:collapse;width:100%;margin-bottom:20px;font-size:13px}
    th{background:#f2ebe0;color:#5a4535;padding:10px 14px;text-align:left;border-bottom:2px solid #d8cdb8;font-size:11px;text-transform:uppercase;letter-spacing:.8px}
    td{padding:9px 14px;border-bottom:1px solid #e8dece;color:#5a4535}
    tr:nth-child(even) td{background:#faf6f0}
    .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700}
    .badge.orange{background:#fff3ea;color:#e8620a}
    .badge.red{background:#fff0ef;color:#c0160f}
    .badge.green{background:#eaf6ef;color:#1a7a3c}
    .badge.ask{background:#eaf1fc;color:#1a4a8a}
    .sug{padding:12px 16px;border-radius:6px;margin-bottom:10px;border-left:4px solid;font-size:13px}
    .sug.high{background:#fff0ef;border-color:#c0160f;color:#9a100a}
    .sug.medium{background:#fdf5dc;border-color:#b87a10;color:#7a5010}
    .sug.low{background:#eaf1fc;border-color:#1a4a8a;color:#0f3268}
    .sug-title{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}
    .step{display:flex;gap:12px;align-items:center;padding:10px 14px;background:#f2ebe0;border-radius:6px;margin-bottom:8px;font-size:13px}
    .step-num{width:26px;height:26px;border-radius:50%;background:#e8620a;color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0}
    .split-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px}
    .split-card{text-align:center;background:#fff;border:1px solid #d8cdb8;border-radius:8px;padding:16px}
    .split-num{font-family:'Playfair Display',serif;font-size:32px;font-weight:900;line-height:1}
    .split-label{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;margin-top:4px;color:#8a7862}
    footer{margin-top:48px;padding-top:20px;border-top:1px solid #d8cdb8;color:#8a7862;font-size:12px;text-align:center}
  </style>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;600&display=swap" rel="stylesheet"/>
</head>
<body>
  <h1>DataPrep Engine</h1>
  <div class="meta">Full Analysis Report &nbsp;·&nbsp; ${p.filename} &nbsp;·&nbsp; Generated ${now}</div>

  <h2>Data Quality Score</h2>
  <div class="score-box">
    <div class="score-main">
      <div class="score-num">${Math.round(qs.overall)}</div>
      <div class="score-label">${qs.rating}</div>
    </div>
    <div class="score-bars">
      ${['completeness','consistency','validity'].map(k => `
        <div class="bar-row">
          <div class="bar-lbl">${k.charAt(0).toUpperCase()+k.slice(1)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.min(qs[k],100)}%"></div></div>
          <div class="bar-val">${Math.round(qs[k])}%</div>
        </div>`).join('')}
    </div>
  </div>

  <h2>Dataset Overview</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>File</td><td>${p.filename}</td></tr>
      <tr><td>Rows</td><td>${p.basic_info.rows.toLocaleString()}</td></tr>
      <tr><td>Columns</td><td>${p.basic_info.columns}</td></tr>
      <tr><td>Memory Usage</td><td>${p.basic_info.memory_usage}</td></tr>
      <tr><td>Missing Data</td><td>${p.missing_values.missing_percentage.toFixed(2)}%</td></tr>
      <tr><td>Duplicate Rows</td><td>${p.basic_info.duplicate_rows}</td></tr>
      <tr><td>ML Problem Type</td><td>${p.ml_problem_type?.suggested_type || '—'}</td></tr>
    </tbody>
  </table>

  <h2>Column Analysis</h2>
  <table>
    <thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Missing %</th><th>Unique</th><th>Flag</th></tr></thead>
    <tbody>
      ${(p.column_analysis || []).map(col => `
        <tr>
          <td><code>${col.name}</code></td>
          <td>${col.dtype}</td>
          <td>${col.missing_count}</td>
          <td>${col.missing_percentage.toFixed(1)}%</td>
          <td>${col.unique_values}</td>
          <td>${col.is_constant ? '<span class="badge orange">CONST</span>' : col.is_identifier ? '<span class="badge red">ID</span>' : '<span class="badge green">OK</span>'}</td>
        </tr>`).join('')}
    </tbody>
  </table>

  <h2>AI Recommendations</h2>
  <p style="margin-bottom:16px;color:#5a4535">${r.overall_assessment}</p>

  <h3>Key Suggestions</h3>
  ${(r.suggestions || []).map(s => `
    <div class="sug ${s.severity}">
      <div class="sug-title">${s.severity.toUpperCase()}</div>
      ${s.text}
      ${s.impact ? `<div style="margin-top:4px;font-size:11px;opacity:.75">Impact: ${s.impact}</div>` : ''}
    </div>`).join('')}

  <h2>Preprocessing Pipeline</h2>
  ${(r.preprocessing_steps || []).map((s, i) => `
    <div class="step"><div class="step-num">${i+1}</div><div>${s}</div></div>`).join('')}

  ${pr ? `
  <h2>Output Results</h2>
  <div class="split-grid">
    <div class="split-card">
      <div class="split-num" style="color:#e8620a">${pr.shapes.train?.[0]}</div>
      <div style="font-size:11px;color:#8a7862;margin-top:2px">${pr.shapes.train?.[1]} features</div>
      <div class="split-label">Train Set</div>
    </div>
    <div class="split-card">
      <div class="split-num" style="color:#1a4a8a">${pr.shapes.val?.[0]}</div>
      <div style="font-size:11px;color:#8a7862;margin-top:2px">${pr.shapes.val?.[1]} features</div>
      <div class="split-label">Validation Set</div>
    </div>
    <div class="split-card">
      <div class="split-num" style="color:#1a7a3c">${pr.shapes.test?.[0]}</div>
      <div style="font-size:11px;color:#8a7862;margin-top:2px">${pr.shapes.test?.[1]} features</div>
      <div class="split-label">Test Set</div>
    </div>
  </div>
  <h3>Processing Log</h3>
  <table>
    <thead><tr><th>#</th><th>Operation</th></tr></thead>
    <tbody>${(pr.log || []).map((l, i) => `<tr><td>${i+1}</td><td>${l}</td></tr>`).join('')}</tbody>
  </table>` : ''}

  <footer>DataPrep Engine &nbsp;·&nbsp; AI-Powered Preprocessing Suite &nbsp;·&nbsp; Report generated ${now}</footer>
</body>
</html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `dataprep_report_${Date.now()}.html`;
  a.click();
  showToast('Report downloaded!', 'success');
}

// ═══════════════════════════════════════════
//  TOAST NOTIFICATIONS
// ═══════════════════════════════════════════
function showToast(message, type = 'info', duration = 3500) {
  const container = document.getElementById('toastContainer');
  const icons = { success:'✓', error:'✕', info:'ℹ', warning:'⚠' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type]||'ℹ'}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = 'toast-out .3s ease forwards';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ═══════════════════════════════════════════
//  OVERLAY LOADER (utility)
// ═══════════════════════════════════════════
function showOverlay(text = 'Processing...') {
  document.getElementById('olText').textContent = text;
  document.getElementById('overlayLoader').style.display = 'flex';
}
function hideOverlay() {
  document.getElementById('overlayLoader').style.display = 'none';
}