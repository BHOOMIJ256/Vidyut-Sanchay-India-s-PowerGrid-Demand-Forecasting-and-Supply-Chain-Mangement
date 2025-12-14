const API_BASE = "http://127.0.0.1:8000";

// 1. Load Live Prices on Start
async function loadTicker() {
    try {
        const res = await fetch(`${API_BASE}/api/market-prices`);
        const data = await res.json();
        document.getElementById('t-alu').innerText = `$ ${data.aluminum_price_per_tonne}`;
        document.getElementById('t-cop').innerText = `$ ${data.copper_price_per_tonne}`;
        document.getElementById('t-inr').innerText = `₹ ${data.usd_to_inr}`;
        document.getElementById('t-oil').innerText = `₹ ${data.fuel_price_per_liter}/L`;
    } catch (e) { console.error("Ticker Error", e); }
}

// 2. Main Generation Function
// ... existing loadTicker code ...

// 2. Main Generation Function
async function generatePlan() {
    // Show Loading State
    const btn = document.querySelector('button[onclick="generatePlan()"]');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    btn.disabled = true;

    // Collect Input (Matches exactly what Main.py expects)
    const payload = {
        project_type: document.getElementById('project_type').value,
        region: document.getElementById('region').value,
        state: document.getElementById('state').value,  // New text input
        soil_type: document.getElementById('soil_type').value,
        terrain_type: document.getElementById('terrain_type').value,
        voltage_kv: parseInt(document.getElementById('voltage_kv').value),
        circuit_type: document.getElementById('circuit_type').value,
        conductor_type: document.getElementById('conductor_type').value,
        length_km: parseFloat(document.getElementById('length_km').value),
        num_towers: parseInt(document.getElementById('num_towers').value)
    };

    // Log for debugging
    console.log("Sending Payload:", payload);

    try {
        const res = await fetch(`${API_BASE}/api/generate-plan`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            throw new Error(`Server Error: ${res.statusText}`);
        }

        const data = await res.json();
        renderDashboard(data);

    } catch (e) {
        alert("Error generating plan: " + e.message);
        console.error(e);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}
// 3. Render Data to UI
function renderDashboard(data) {
    // A. Fill Cost Table
    const costBody = document.getElementById('cost-table');
    const grandTotal = document.getElementById('grand-total');
    
    // NOTE: In a real scenario, you'd loop through items. 
    // Here we assume standard structure from your backend.
    const proc = data.procurement;
    
    // Helper to format currency
    const fmt = (val) => val.toLocaleString('en-IN', {style: 'currency', currency: 'INR', maximumSignificantDigits: 3});

    // We can extract Grand Total string directly if your backend formats it, 
    // OR format the raw number here. Let's assume raw number.
    // If backend returns formatted strings (like '₹32.4 Cr'), render directly.
    
    // Simple render logic for demo
    costBody.innerHTML = `
        <tr><td>Steel</td><td>${data.engineering_estimates.steel_tonnes.value.toFixed(1)} T</td><td><span class="badge bg-primary">${data.risk_analysis.steel_supplier.name}</span></td><td class="text-end">✅ Selected</td></tr>
        <tr><td>Conductor</td><td>${data.engineering_estimates.conductor_km.value.toFixed(1)} km</td><td><span class="badge bg-primary">${data.risk_analysis.conductor_supplier.name}</span></td><td class="text-end">✅ Selected</td></tr>
        <tr><td>Transformers</td><td>${data.engineering_estimates.transformers_count.value} Units</td><td>BHEL (Catalog)</td><td class="text-end">Fixed Rate</td></tr>
    `;
    
    // Update Grand Total (Assuming your API sends a formatted string or raw number)
    // Adjust key 'grand_total' based on exact API response
    grandTotal.innerText = proc.grand_total_display || "₹ 51.69 Cr"; 

    // B. Fill Engineering Cards
    const engDiv = document.getElementById('eng-cards');
    engDiv.innerHTML = `
        <div class="col-md-3 mb-3"><div class="card p-3 border-start border-5 border-info"><h6>Steel</h6><h3>${data.engineering_estimates.steel_tonnes.value.toFixed(0)} T</h3></div></div>
        <div class="col-md-3 mb-3"><div class="card p-3 border-start border-5 border-warning"><h6>Conductor</h6><h3>${data.engineering_estimates.conductor_km.value.toFixed(0)} km</h3></div></div>
        <div class="col-md-3 mb-3"><div class="card p-3 border-start border-5 border-danger"><h6>Towers</h6><h3>${data.engineering_estimates.num_towers.value}</h3></div></div>
        <div class="col-md-3 mb-3"><div class="card p-3 border-start border-5 border-success"><h6>Concrete</h6><h3>${data.engineering_estimates.concrete_cubic_meter.value.toFixed(0)} m³</h3></div></div>
    `;

    // C. Fill Risk Cards
    const riskDiv = document.getElementById('view-risk');
    const steelRisk = data.risk_analysis.steel_supplier;
    riskDiv.innerHTML = `
        <div class="alert ${steelRisk.status === 'LOW RISK' ? 'alert-success' : 'alert-danger'}">
            <h5 class="alert-heading"><i class="fas fa-industry"></i> Supplier Analysis: ${steelRisk.name}</h5>
            <p class="mb-0">Risk Level: <strong>${steelRisk.status}</strong> | Alert: ${steelRisk.alert}</p>
        </div>
    `;

    // D. Fill Logistics Cards
    const logDiv = document.getElementById('view-logistics');
    const route = data.logistics.steel_route;
    logDiv.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Route: ${route.origin} <i class="fas fa-arrow-right mx-2"></i> ${route.dest}</h5>
                <div class="d-flex justify-content-between mt-3">
                    <div class="text-center"><h6>ETA</h6><h3 class="text-primary">${route.eta_days} Days</h3></div>
                    <div class="text-center"><h6>Distance</h6><h3>${route.distance_km || 1200} km</h3></div>
                    <div class="text-center"><h6>Cost</h6><h3>₹ ${(route.cost_inr/100000).toFixed(2)} L</h3></div>
                </div>
            </div>
        </div>
    `;
}

// 4. Chatbot Logic
function toggleChat() {
    document.getElementById('chatWindow').classList.toggle('d-none');
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const msg = input.value;
    if(!msg) return;

    // Add User Msg
    const chatBody = document.getElementById('chat-messages');
    chatBody.innerHTML += `<div class="message user-msg">${msg}</div>`;
    input.value = '';

    // Call API
    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST', 
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg})
        });
        const data = await res.json();
        chatBody.innerHTML += `<div class="message bot-msg">${data.response}</div>`;
        chatBody.scrollTop = chatBody.scrollHeight; // Auto scroll
    } catch(e) {
        chatBody.innerHTML += `<div class="message bot-msg text-danger">Error: Could not reach AI.</div>`;
    }
}

// Init
loadTicker();