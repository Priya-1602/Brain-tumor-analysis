// Manual flip view toggle
const flipBtn = document.getElementById('flip-btn');
if (flipBtn) {
  flipBtn.addEventListener('click', () => {
    const cur = gradcamImg.style.transform || '';
    const hasFlip = cur.includes('scaleX(-1)');
    const hasRotate = cur.includes('rotate(180deg)');
    const nextFlip = hasFlip ? '' : ' scaleX(-1)';
    const rotatePart = hasRotate ? ' rotate(180deg)' : '';
    gradcamImg.style.transform = (nextFlip + rotatePart).trim();
  });
}

const form = document.getElementById('upload-form');
const statusEl = document.getElementById('status');
const predEl = document.getElementById('prediction');
const confEl = document.getElementById('confidence');
const diagEl = document.getElementById('diagnostic');
const treatEl = document.getElementById('treatment');
const gradcamImg = document.getElementById('gradcam');
const downloadBtn = document.getElementById('download-btn');
let lastResult = null;

function setLoading(isLoading) {
  const btn = document.getElementById('submit-btn');
  if (isLoading) {
    btn.disabled = true;
    btn.textContent = 'Analyzing…';
    statusEl.textContent = 'Running ViT model and Grad-CAM…';
  } else {
    btn.disabled = false;
    btn.textContent = 'Analyze';
    statusEl.textContent = '';
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('image');
  if (!fileInput.files || fileInput.files.length === 0) {
    statusEl.textContent = 'Please select an image first.';
    return;
  }

  const formData = new FormData();
  formData.append('image', fileInput.files[0]);

  try {
    setLoading(true);
    // Timeout after 90 seconds to avoid infinite waiting
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000);
    const res = await fetch('/predict', { method: 'POST', body: formData, signal: controller.signal });
    clearTimeout(timeoutId);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');

    // Update UI
    predEl.textContent = data.prediction ?? '—';
    const conf = (data.confidence ?? 0) * 100;
    confEl.textContent = `${conf.toFixed(1)}%`;
    // Risk assessment
    const riskBadge = document.getElementById('risk-badge');
    const riskNote = document.getElementById('risk-note');
    const riskFill = document.getElementById('risk-fill');
    const riskPercentEl = document.getElementById('risk-percent');
    const riskLabelEl = document.getElementById('risk-label');
    const riskSection = document.getElementById('risk-section');
    const label = (data.prediction || '').toLowerCase();
    const c = (data.confidence || 0);
    let risk = 'SAFE';
    let note = 'No immediate concern.';
    if (label === 'no_tumor') {
      risk = 'SAFE';
      note = 'Model did not detect a tumor.';
    } else {
      if (c >= 0.80) { risk = 'HIGH'; note = 'High likelihood of tumor; expedite specialist review.'; }
      else if (c >= 0.50) { risk = 'MEDIUM'; note = 'Possible tumor; recommend clinical correlation and further evaluation.'; }
      else { risk = 'SAFE'; note = 'Low confidence; consider repeat imaging if symptoms persist.'; }
    }
    if (riskBadge) {
      riskBadge.textContent = risk;
      riskBadge.classList.remove('pill-safe','pill-medium','pill-high');
      riskBadge.classList.add(risk === 'HIGH' ? 'pill-high' : risk === 'MEDIUM' ? 'pill-medium' : 'pill-safe');
    }
    if (riskNote) riskNote.textContent = note;
    // Update progress meter
    if (riskFill) {
      const pct = Math.max(0, Math.min(100, conf));
      riskFill.style.width = pct.toFixed(0) + '%';
      // Color by risk
      if (risk === 'HIGH') riskFill.style.background = 'linear-gradient(90deg, #fecaca, #ef4444)';
      else if (risk === 'MEDIUM') riskFill.style.background = 'linear-gradient(90deg, #fde68a, #f59e0b)';
      else riskFill.style.background = 'linear-gradient(90deg, #bbf7d0, #22c55e)';
    }
    if (riskPercentEl) riskPercentEl.textContent = `${conf.toFixed(0)}%`;
    if (riskLabelEl) riskLabelEl.textContent = `Risk level — ${risk}`;
    if (riskSection) riskSection.style.display = 'block';
    diagEl.textContent = data.diagnostic_text ?? '—';
    const locEl = document.getElementById('localization');
    locEl.textContent = data.localization ? `Estimated location: ${data.localization}` : '—';
    treatEl.textContent = data.treatment_suggestion ?? '—';

    const isNoTumor = (data.prediction || '').toLowerCase() === 'no_tumor';
    if (!isNoTumor && data.gradcam_url) {
      gradcamImg.src = data.gradcam_url + `?t=${Date.now()}`; // bust cache
      gradcamImg.alt = `Grad-CAM overlay for ${data.prediction}`;
      gradcamImg.style.display = 'block';
      // Ensure no auto-rotation is applied; reset any rotate
      const curT = gradcamImg.style.transform || '';
      gradcamImg.style.transform = curT.replace('rotate(180deg)','').trim();
    } else {
      // No tumor or no overlay: hide and clear the Grad-CAM image
      gradcamImg.removeAttribute('src');
      gradcamImg.style.display = 'none';
      gradcamImg.style.transform = '';
    }
    
    // Show download button after successful analysis
    const downloadBtn = document.getElementById('download-btn');
    if (downloadBtn) downloadBtn.style.display = 'block';
    // Optionally, could display mask url or bbox later

    // Render highlight chips
    const h = document.getElementById('highlights');
    h.innerHTML = '';
    (data.highlights || []).forEach(tok => {
      const span = document.createElement('span');
      span.className = 'pill ' + (tok.type === 'anat' ? 'pill-ana' : tok.type === 'proc' ? 'pill-proc' : 'pill-primary');
      span.textContent = tok.text;
      h.appendChild(span);
    });

    // ----- Charts: Lab Report vs My Report (Pie) -----
    try {
      const chartsSection = document.getElementById('charts-section');
      const labCtx = document.getElementById('labChart')?.getContext('2d');
      const myCtx = document.getElementById('myChart')?.getContext('2d');
      if (chartsSection && labCtx && myCtx && window.Chart) {
        chartsSection.style.display = 'block';

        // Dynamic-ish weights can be lightly influenced by confidence
        const confPct = Math.max(0, Math.min(100, Math.round((data.confidence || 0) * 100)));

        const labLabels = [
          'Histopathology/Markers',
          'Radiologist Impression',
          'Lesion Metrics (size/site)',
          'Differential Dx',
          'Follow‑up Protocol'
        ];
        const labData = [24, 20, 20, 18, 18];

        const myLabels = [
          'ViT‑B/16 Logits (4‑class)',
          'Softmax Confidence',
          'Risk Stratification',
          'Lobe/Hemisphere Inference',
          'Grad‑CAM Saliency Coverage',
          'Patient‑friendly Narrative (PDF)',
          'Patient Metadata (Name/Phone)',
          'LLM Clinical Rationale'
        ];
        const myData = [16, 18, 16, 14, 14, 10, 6, 6];
        
        const palette = ['#60a5fa','#34d399','#f59e0b','#ef4444','#a78bfa','#22d3ee','#fb7185','#10b981'];

        // Recreate or update charts
        if (window._labChart) { window._labChart.destroy(); }
        if (window._myChart) { window._myChart.destroy(); }

        window._labChart = new Chart(labCtx, {
          type: 'pie',
          data: { labels: labLabels, datasets: [{ data: labData, backgroundColor: palette }] },
          options: { responsive: false, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
        });

        window._myChart = new Chart(myCtx, {
          type: 'pie',
          data: { labels: myLabels, datasets: [{ data: myData, backgroundColor: palette }] },
          options: { responsive: false, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
        });
      }
    } catch (e) { /* noop */ }

    // Store last result for report generation (including original_url)
    lastResult = data;
  } catch (err) {
    console.error(err);
    if (err.name === 'AbortError') {
      statusEl.textContent = 'Request timed out. The model may be loading; please retry.';
    } else {
      statusEl.textContent = `Error: ${err.message}`;
    }
  } finally {
    setLoading(false);
  }
});

downloadBtn.addEventListener('click', async () => {
  if (!lastResult) {
    alert('Please analyze an image first');
    return;
  }
  // Prompt for patient details (no UI changes required)
  const patientName = (window.prompt('Enter patient name:') || '').trim();
  if (!patientName) {
    alert('Patient name is required to generate the report.');
    return;
  }
  const patientPhone = (window.prompt('Enter patient phone number:') || '').trim();
  if (!patientPhone) {
    alert('Patient phone number is required to generate the report.');
    return;
  }
  // Build payload from current UI state
  const payload = {
    prediction: predEl.textContent || '',
    confidence: parseFloat((confEl.textContent || '0').replace('%',''))/100 || 0,
    localization: (document.getElementById('localization').textContent || '').replace('Estimated location: ','') || '',
    diagnostic_text: diagEl.textContent || '',
    treatment_suggestion: treatEl.textContent || '',
    gradcam_url: lastResult && lastResult.gradcam_url ? lastResult.gradcam_url : (gradcamImg && gradcamImg.src && gradcamImg.src.includes('/static/') ? gradcamImg.src.split(location.origin)[1] : null),
    original_url: lastResult && lastResult.original_url ? lastResult.original_url : null,
    patient_name: patientName,
    patient_phone: patientPhone,
  };

  try {
    const res = await fetch('/report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to generate report');
    if (data.report_url) {
      const a = document.createElement('a');
      a.href = data.report_url + `?t=${Date.now()}`;
      a.download = 'brain_tumor_report.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Report error: ${err.message}`;
  }
});


