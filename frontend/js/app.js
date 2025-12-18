const API_BASE = "http://127.0.0.1:8000";

// --- GLOBAL VARIABLE FOR BOT MEMORY ---
let currentDashboardData = null; 

// 1. Load Live Prices on Start
async function loadTicker() {
    try {
        const res = await fetch(`${API_BASE}/api/market-prices`);
        if (!res.ok) throw new Error("Ticker Failed");
        const data = await res.json();
        
        document.getElementById('t-alu').innerText = `$ ${data.aluminum_price_per_tonne || '2500'}`;
        document.getElementById('t-cop').innerText = `$ ${data.copper_price_per_tonne || '9800'}`;
        document.getElementById('t-inr').innerText = `₹ ${data.usd_to_inr || '84.0'}`;
        document.getElementById('t-oil').innerText = `₹ ${data.fuel_price_per_liter || '90.0'}/L`;
    } catch (e) { console.error("Ticker Error", e); }
}

// 2. Main Generation Function
async function generatePlan() {
    // UI Feedback
    const btn = document.querySelector('button[onclick="generatePlan()"]');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing (AI Agents Active)...';
    btn.disabled = true;

    // Collect Input
    const payload = {
        project_type: document.getElementById('project_type').value,
        region: document.getElementById('region').value,
        project_city: document.getElementById('project_city').value,
        soil_type: document.getElementById('soil_type').value,
        terrain_type: document.getElementById('terrain_type').value,
        voltage_kv: parseInt(document.getElementById('voltage_kv').value) || 132,
        circuit_type: document.getElementById('circuit_type').value,
        conductor_type: document.getElementById('conductor_type').value,
        length_km: parseFloat(document.getElementById('length_km').value) || 71.79,
        num_towers: parseInt(document.getElementById('num_towers').value) || 282
    };

    try {
        console.log("🚀 Sending Request:", payload);
        const res = await fetch(`${API_BASE}/api/generate-plan`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(`Server Error: ${res.status}`);

        const data = await res.json();
        console.log("✅ Data Received:", data); 
        
        // --- CRITICAL FIX: Save data for the Chatbot ---
        currentDashboardData = data; 
        
        renderDashboard(data);

    } catch (e) {
        alert("Error: " + e.message);
        console.error(e);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// 3. Render Data (The Critical Part)
function renderDashboard(data) {
    // --- A. COST TABLE ---
    const proc = data.procurement;
    const est = data.engineering_estimates;
    
    // Formatting Helper
    const fmt = (num) => num ? num.toLocaleString('en-IN') : '0';
    const currency = (num) => "₹ " + (num ? (num/10000000).toFixed(2) + " Cr" : "0.00 Cr");

    // Update Grand Total
    const grandTotalVal = proc.grand_total || 0;
    document.getElementById('grand-total').innerText = currency(grandTotalVal);

    // Populate Table
    const costBody = document.getElementById('cost-table');
    costBody.innerHTML = `
        <tr>
            <td><i class="fas fa-hammer text-muted me-2"></i>Steel (Towers)</td>
            <td>${fmt(est.steel_tonnes.value)} T</td>
            <td><span class="badge bg-primary">${proc.steel_supplier || 'Unknown'}</span></td>
            <td class="text-end fw-bold">Selected</td>
        </tr>
        <tr>
            <td><i class="fas fa-bolt text-muted me-2"></i>Conductor</td>
            <td>${fmt(est.conductor_km.value)} km</td>
            <td><span class="badge bg-info text-dark">GUPTA_PWR</span></td>
            <td class="text-end fw-bold">Selected</td>
        </tr>
        <tr>
            <td><i class="fas fa-cubes text-muted me-2"></i>Concrete</td>
            <td>${fmt(est.concrete_cubic_meter.value)} m³</td>
            <td>LOCAL_MIX</td>
            <td class="text-end">Selected</td>
        </tr>
        <tr>
            <td><i class="fas fa-ring text-muted me-2"></i>Insulators</td>
            <td>${fmt(est.insulators_unit.value)} Units</td>
            <td>Base Rate</td>
            <td class="text-end">Standard</td>
        </tr>
    `;

    // --- B. ENGINEERING CARDS ---
    const engDiv = document.getElementById('eng-cards');
    engDiv.innerHTML = `
        <div class="col-md-3 mb-3"><div class="card p-3 border-start border-5 border-info shadow-sm">
            <h6 class="text-muted small">STEEL REQUIRED</h6><h3>${fmt(est.steel_tonnes.value)} <small class="text-muted fs-6">T</small></h3>
        </div></div>
        <div class="col-md-3 mb-3"><div class="card p-3 border-start border-5 border-warning shadow-sm">
            <h6 class="text-muted small">CONDUCTOR LEN</h6><h3>${fmt(est.conductor_km.value)} <small class="text-muted fs-6">km</small></h3>
        </div></div>
        <div class="col-md-3 mb-3"><div class="card p-3 border-start border-5 border-danger shadow-sm">
            <h6 class="text-muted small">TOWERS</h6><h3>${est.num_towers.value} <small class="text-muted fs-6">Nos</small></h3>
        </div></div>
        <div class="col-md-3 mb-3"><div class="card p-3 border-start border-5 border-success shadow-sm">
            <h6 class="text-muted small">CONCRETE</h6><h3>${fmt(est.concrete_cubic_meter.value)} <small class="text-muted fs-6">m³</small></h3>
        </div></div>
    `;

    // --- C. LOGISTICS (Fixing the path issue) ---
    // --- C. LOGISTICS (Now Handles Multiple Routes) ---
    const logDiv = document.getElementById('logistics-cards');
    const routes = data.logistics.routes || []; // Expecting a list now

    if (routes.length > 0) {
        let html = '';
        routes.forEach(route => {
            html += `
            <div class="col-12 mb-3">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <div class="d-flex align-items-center justify-content-between mb-4">
                            <div>
                                <h6 class="text-muted mb-1">SUPPLIER</h6>
                                <h4 class="fw-bold text-primary"><i class="fas fa-warehouse me-2"></i>${route.origin_supplier}</h4>
                            </div>
                            <div class="text-center px-4">
                                <div class="text-muted small mb-1">${route.distance_km ? route.distance_km.toFixed(0) : 0} km</div>
                                <i class="fas fa-arrow-right fa-2x text-muted opacity-25"></i>
                            </div>
                            <div class="text-end">
                                <h6 class="text-muted mb-1">SITE</h6>
                                <h4 class="fw-bold text-success">${route.destination_project}<i class="fas fa-map-marker-alt ms-2"></i></h4>
                            </div>
                        </div>
                        
                        <div class="row g-3 text-center">
                            <div class="col-4 border-end">
                                <h6 class="text-muted small">ETA</h6>
                                <h3 class="text-dark-blue">${route.transit_time_days ? route.transit_time_days.toFixed(1) : 0} Days</h3>
                            </div>
                            <div class="col-4 border-end">
                                <h6 class="text-muted small">ARRIVAL</h6>
                                <h3 class="text-dark-blue">${route.est_arrival_date || 'N/A'}</h3>
                            </div>
                            <div class="col-4">
                                <h6 class="text-muted small">COST</h6>
                                <h3 class="text-dark-blue">₹ ${(route.transport_cost_inr / 100000).toFixed(2)} L</h3>
                            </div>
                        </div>
                    </div>
                </div>
            </div>`;
        });
        logDiv.innerHTML = html;
    } else {
        logDiv.innerHTML = `<div class="col-12 text-center text-muted">Logistics Data Unavailable</div>`;
    }

    // --- D. RISK ANALYSIS ---
    // --- D. RISK ANALYSIS (Multi-Card) ---
    const riskDiv = document.getElementById('risk-cards');
    // NOTE: Backend now sends 'reports' which is a list
    const riskReports = data.risk_analysis.reports || [];
    
    if (riskReports.length > 0) {
        let html = '';
        riskReports.forEach(report => {
            // Color Logic
            let alertClass = 'alert-success'; // Green
            let icon = 'fa-check-circle';
            
            if (report.risk_score > 7) {
                alertClass = 'alert-danger'; // Red
                icon = 'fa-exclamation-triangle';
            } else if (report.risk_score > 4) {
                alertClass = 'alert-warning'; // Yellow
                icon = 'fa-exclamation-circle';
            }

            html += `
            <div class="col-12 mb-3">
                <div class="alert ${alertClass} shadow-sm border-0">
                    <div class="d-flex align-items-start">
                        <div class="me-3 mt-1"><i class="fas ${icon} fa-2x"></i></div>
                        <div class="w-100">
                            <div class="d-flex justify-content-between">
                                <h5 class="alert-heading fw-bold">${report.company}</h5>
                                <span class="badge bg-white text-dark border">Score: ${report.risk_score}/10</span>
                            </div>
                            <p class="mb-1"><strong>Reason:</strong> ${report.reason || report.alert || 'Analysis pending'}</p>
                        </div>
                    </div>
                </div>
            </div>`;
        });
        riskDiv.innerHTML = html;
    } else {
        riskDiv.innerHTML = `<div class="col-12 text-center text-muted">Risk Data Unavailable</div>`;
    }
}

// ... Chatbot Code remains same ...
// Init
loadTicker();

// 4. Chatbot Logic
// ================= CHATBOT LOGIC =================

// 1. Toggle Window Visibility
function toggleChat() {
    const chatWindow = document.getElementById('chat-window');
    const chatBtn = document.getElementById('chat-launcher');
    
    // Toggle the 'd-none' (Bootstrap Display None) class
    if (chatWindow.classList.contains('d-none')) {
        chatWindow.classList.remove('d-none');
        chatBtn.style.transform = 'scale(0)'; // Hide button
        setTimeout(() => document.getElementById('chat-input').focus(), 100);
    } else {
        chatWindow.classList.add('d-none');
        chatBtn.style.transform = 'scale(1)'; // Show button
    }
}

// 2. Handle Enter Key
function handleEnter(event) {
    if (event.key === 'Enter') sendMessage();
}

// 3. Send Message to Backend
async function sendMessage() {
    const inputField = document.getElementById('chat-input');
    const msg = inputField.value.trim();
    if (!msg) return;

    // A. Show User Message
    addChatMessage(msg, 'user');
    inputField.value = '';
    
    // B. Show "Thinking..." (With a FIXED ID)
    // We remove any existing loader first, just in case
    removeLoader(); 
    addChatMessage("Thinking...", 'bot', true); // This will now create ID="bot-loading-indicator"

    // C. Prepare Context
    const context = (typeof currentDashboardData !== 'undefined') ? currentDashboardData : null;

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                message: msg,
                context: context 
            })
        });

        const data = await res.json();
        
        // D. Remove Loader & Show Response
        removeLoader();
        addChatMessage(data.response, 'bot');

    } catch (e) {
        removeLoader();
        addChatMessage("⚠️ Error: Agent offline.", 'bot');
        console.error(e);
    }
}

// Helper to remove loader reliably
function removeLoader() {
    const loader = document.getElementById('bot-loading-indicator');
    if (loader) loader.remove();
}

// 4. Helper to Append Messages (Updated)
function addChatMessage(text, sender, isTyping = false) {
    const chatBody = document.getElementById('chat-messages');
    const div = document.createElement('div');
    
    if (isTyping) {
        // FIXED ID for the loader so we can always find it
        div.id = 'bot-loading-indicator';
        div.className = 'message-bot text-muted fst-italic';
        div.innerHTML = `<i class="fas fa-circle-notch fa-spin text-primary me-2"></i> Thinking...`;
    } else {
        // Unique ID for normal messages
        div.id = 'msg-' + Date.now();
        
        if (sender === 'user') {
            div.className = 'message-user';
            div.innerText = text;
        } else {
            div.className = 'message-bot';
            div.innerHTML = formatBotResponse(text);
        }
    }
    
    chatBody.appendChild(div);
    chatBody.scrollTop = chatBody.scrollHeight; // Auto-scroll
    return div.id;
}

// 5. Simple Markdown Formatter for Bot (Bold, Lists)
function formatBotResponse(text) {
    let clean = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
        .replace(/\n/g, '<br>'); // Newlines
    return clean;
}

// ================= REPORT GENERATOR =================

function downloadReport() {
    // 1. Check if data exists
    if (!currentDashboardData) {
        alert("⚠️ No plan generated yet. Please click 'Generate Plan' first.");
        return;
    }

    const data = currentDashboardData;
    const est = data.engineering_estimates;
    const proc = data.procurement;
    const routes = data.logistics.routes || [];
    const risks = data.risk_analysis.reports || [];

    // 2. Build the Text Report
    let report = `==========================================================
⚡ VIDYUT SANCHAY - PROCUREMENT ANALYSIS REPORT
==========================================================
Date Generated: ${new Date().toLocaleString()}
Project City:   ${document.getElementById('project_city').value}
Project Type:   ${document.getElementById('project_type').value}
----------------------------------------------------------

1. FINANCIAL SUMMARY
--------------------
GRAND TOTAL:    ₹ ${(proc.grand_total / 10000000).toFixed(2)} Cr
Steel Cost:     ₹ ${(proc.steel_cost / 10000000).toFixed(2)} Cr
Supplier:       ${proc.steel_supplier}

2. ENGINEERING SPECS
--------------------
Steel Required: ${est.steel_tonnes.value.toLocaleString()} Tonnes
Conductor Len:  ${est.conductor_km.value.toLocaleString()} km
Towers Count:   ${est.num_towers.value} units
Concrete Vol:   ${est.concrete_cubic_meter.value.toLocaleString()} m3

3. LOGISTICS PLAN
-----------------
`;

    // Loop through routes
    routes.forEach((r, index) => {
        report += `
   [Route ${index + 1}] ${r.origin_supplier} -> ${r.destination_project}
   - Distance:    ${r.distance_km} km
   - ETA:         ${r.transit_time_days.toFixed(1)} Days
   - Est Arrival: ${r.est_arrival_date}
   - Cost:        ₹ ${(r.transport_cost_inr / 100000).toFixed(2)} Lakhs
`;
    });

    report += `
4. RISK ANALYSIS
----------------
`;

    // Loop through risks
    risks.forEach((r) => {
        report += `
   [${r.company}] Score: ${r.risk_score}/10
   - Verdict: ${r.reason || 'No specific alert.'}
`;
    });

    report += `
==========================================================
Generated by Vidyut Sanchay AI Orchestrator
==========================================================
`;

    // 3. Trigger Download
    const blob = new Blob([report], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `Vidyut_Report_${Date.now()}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}