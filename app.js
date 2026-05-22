// Global Application State
let df_nat = [];
let df_states = [];
let latestRow = {};
let prevYearRow = {};

// ML Model Coefficients (Extracted from training pipeline)
const FORECAST_MODEL = {
    intercept: 57456.252269611316,
    trend: 422.0073210591706,
    monthCoefs: {
        2: -6922.9111039991785,
        3: 7367.882668731328,
        4: -2054.938511651596,
        5: 5230.155192272349,
        6: 2180.67623569407,
        7: 4354.195243879355,
        8: 3303.144790884779,
        9: -1874.4668792625953,
        10: -1143.4956874089328,
        11: -11224.103556333843,
        12: 901.7451369427741
    },
    lag1: 0.5986277588092375,
    lag2: -0.18162109855115116,
    lag12: 0.014283141361927295
};

const EMISSIONS_MODEL = {
    intercept: 230.03798585599822,
    coalShare: 589.6250726271209,
    gasShare: 265.16833898727316,
    solarShare: -182.69795194869928,
    windShare: -218.78724344475538,
    hydroShare: -206.79770499561766,
    nuclearShare: -216.59558902692467
};

// Document Elements
document.addEventListener("DOMContentLoaded", () => {
    // Navigation routing
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            navItems.forEach(n => n.classList.remove("active"));
            item.classList.add("active");
            
            const targetPage = item.getAttribute("data-page");
            showPage(targetPage);
        });
    });

    // Start data load
    loadData();
});

// Switch pages
function showPage(pageId) {
    document.querySelectorAll(".page-section").forEach(sec => {
        sec.classList.remove("active");
    });
    const activeSection = document.getElementById(pageId);
    activeSection.classList.add("active");

    // Trigger page-specific rendering
    if (pageId === "page-national") {
        renderNationalDashboard();
    } else if (pageId === "page-state") {
        renderStateExplorer();
    } else if (pageId === "page-ml") {
        renderMLForecastSimulator();
    }
}

// Fetch and parse datasets
async function loadData() {
    try {
        const [resNat, resStates] = await Promise.all([
            fetch("pivoted_india_total_monthly.csv"),
            fetch("pivoted_states_monthly.csv")
        ]);

        const csvNat = await resNat.text();
        const csvStates = await resStates.text();

        // Parse pivoted_india_total_monthly.csv
        Papa.parse(csvNat, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                df_nat = results.data;
                // Parse pivoted_states_monthly.csv
                Papa.parse(csvStates, {
                    header: true,
                    dynamicTyping: true,
                    skipEmptyLines: true,
                    complete: (resultsStates) => {
                        df_states = resultsStates.data;
                        
                        // Sort by Date (using string comparison which is perfect for YYYY-MM-DD format)
                        df_nat.sort((a, b) => a.Date.localeCompare(b.Date));
                        df_states.sort((a, b) => a.Date.localeCompare(b.Date));
                        
                        // Extract latest records
                        latestRow = df_nat[df_nat.length - 1];
                        
                        // Find matching row from one year ago without fragile Date parsing
                        const latestParts = latestRow.Date.split('-');
                        const prevYearStr = `${parseInt(latestParts[0], 10) - 1}-${latestParts[1]}-01`;
                        prevYearRow = df_nat.find(row => row.Date === prevYearStr) || df_nat[df_nat.length - 13] || df_nat[0];

                        // Hide loading overlay
                        const loader = document.getElementById("loading-overlay");
                        loader.style.opacity = 0;
                        setTimeout(() => loader.style.display = "none", 500);

                        // Render default landing page (National Dashboard)
                        showPage("page-national");
                    }
                });
            }
        });
    } catch (err) {
        console.error("Error loading CSV files:", err);
        alert("Failed to load clean CSV files. Make sure they are placed in the same directory.");
    }
}

// Format numbers
function formatNum(val, decimals = 0) {
    if (val === null || val === undefined || isNaN(val)) return "-";
    return val.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

// Render Page 1: National Dashboard
function renderNationalDashboard() {
    // 1. Fill KPI metrics
    document.getElementById("nat-metric-gen").innerHTML = `${formatNum(latestRow.Total_Generation_GWh)} GWh`;
    if (prevYearRow) {
        const diffGen = ((latestRow.Total_Generation_GWh - prevYearRow.Total_Generation_GWh) / prevYearRow.Total_Generation_GWh) * 100;
        document.getElementById("nat-delta-gen").innerHTML = `<i class="fa-solid ${diffGen >= 0 ? 'fa-arrow-trend-up' : 'fa-arrow-trend-down'}"></i> ${diffGen.toFixed(1)}% YoY`;
        document.getElementById("nat-delta-gen").className = `metric-delta ${diffGen >= 0 ? 'delta-up' : 'delta-down'}`;
    }

    document.getElementById("nat-metric-clean").innerHTML = `${(latestRow.Clean_Share * 100).toFixed(1)}%`;
    if (prevYearRow) {
        const diffClean = (latestRow.Clean_Share - prevYearRow.Clean_Share) * 100;
        document.getElementById("nat-delta-clean").innerHTML = `<i class="fa-solid ${diffClean >= 0 ? 'fa-arrow-trend-up' : 'fa-arrow-trend-down'}"></i> ${diffClean.toFixed(1)}% YoY`;
        document.getElementById("nat-delta-clean").className = `metric-delta ${diffClean >= 0 ? 'delta-up' : 'delta-down'}`;
    }

    document.getElementById("nat-metric-co2").innerHTML = `${latestRow.CO2_Intensity_gCO2_kWh.toFixed(1)} g/kWh`;
    if (prevYearRow) {
        const diffCO2 = latestRow.CO2_Intensity_gCO2_kWh - prevYearRow.CO2_Intensity_gCO2_kWh;
        document.getElementById("nat-delta-co2").innerHTML = `<i class="fa-solid ${diffCO2 <= 0 ? 'fa-arrow-trend-down' : 'fa-arrow-trend-up'}"></i> ${diffCO2.toFixed(1)} g/kWh`;
        document.getElementById("nat-delta-co2").className = `metric-delta ${diffCO2 <= 0 ? 'delta-up' : 'delta-down'}`; // Inverse: lower is better
    }

    document.getElementById("nat-metric-em").innerHTML = `${formatNum(latestRow.Total_Emissions_ktCO2)} ktCO2`;
    if (prevYearRow) {
        const diffEm = ((latestRow.Total_Emissions_ktCO2 - prevYearRow.Total_Emissions_ktCO2) / prevYearRow.Total_Emissions_ktCO2) * 100;
        document.getElementById("nat-delta-em").innerHTML = `<i class="fa-solid ${diffEm <= 0 ? 'fa-arrow-trend-down' : 'fa-arrow-trend-up'}"></i> ${diffEm.toFixed(1)}% YoY`;
        document.getElementById("nat-delta-em").className = `metric-delta ${diffEm <= 0 ? 'delta-up' : 'delta-down'}`; // Inverse
    }

    // 2. Capacity Expansion Chart
    const dates = df_nat.map(row => row.Date);
    const traceCapFossil = {
        x: dates,
        y: df_nat.map(row => row.Fossil_Capacity_MW / 1000),
        name: 'Fossil Capacity',
        mode: 'lines',
        line: { color: '#ff5252', width: 3 }
    };
    const traceCapClean = {
        x: dates,
        y: df_nat.map(row => row.Clean_Capacity_MW / 1000),
        name: 'Clean Capacity',
        mode: 'lines',
        line: { color: '#2ecc71', width: 3 }
    };
    
    const layoutCap = {
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 40, r: 10, t: 10, b: 40 },
        xaxis: { title: 'Year', showgrid: false },
        yaxis: { title: 'Capacity (GW)', gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { orientation: 'h', y: 1.1, x: 1, xanchor: 'right' },
        hovermode: 'x unified'
    };
    Plotly.newPlot('chart-capacity', [traceCapFossil, traceCapClean], layoutCap, { responsive: true });

    // 3. Generation Mix Stacked Area Chart
    const fuels = ['Coal_Share', 'Gas_Share', 'Hydro_Share', 'Nuclear_Share', 'Solar_Share', 'Wind_Share', 'Bioenergy_Share'];
    const fuelLabels = ['Coal', 'Gas', 'Hydro', 'Nuclear', 'Solar', 'Wind', 'Bioenergy'];
    const fuelColors = ['#2c3e50', '#e67e22', '#2980b9', '#3498db', '#f1c40f', '#2ecc71', '#8e44ad'];
    
    // Construct cumulative traces for stacked area chart
    const tracesGen = [];
    let cumulative = new Array(df_nat.length).fill(0);
    
    fuels.forEach((fuel, fIdx) => {
        const yVals = df_nat.map((row, rIdx) => {
            cumulative[rIdx] += (row[fuel] || 0) * 100;
            return cumulative[rIdx];
        });
        
        tracesGen.push({
            x: dates,
            y: yVals,
            name: fuelLabels[fIdx],
            fill: 'tonexty',
            mode: 'lines',
            line: { width: 0.5, color: fuelColors[fIdx] },
            fillcolor: fuelColors[fIdx]
        });
    });
    
    // Plotly stacked area traces need to be plotted back-to-front or as stacked variables
    // Simple way is using Plotly's stackgroup:
    const finalTracesGen = fuels.map((fuel, fIdx) => {
        return {
            x: dates,
            y: df_nat.map(row => (row[fuel] || 0) * 100),
            name: fuelLabels[fIdx],
            stackgroup: 'one',
            fillcolor: fuelColors[fIdx],
            line: { color: fuelColors[fIdx], width: 1 }
        };
    });

    const layoutGen = {
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 40, r: 10, t: 10, b: 40 },
        xaxis: { title: 'Year', showgrid: false },
        yaxis: { title: 'Generation Share (%)', range: [0, 100], gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { orientation: 'h', y: 1.1, x: 1, xanchor: 'right' },
        hovermode: 'x unified'
    };
    Plotly.newPlot('chart-generation-mix', finalTracesGen, layoutGen, { responsive: true });

    // 4. Emissions Intensity & Total Emissions dual axis/multi-line chart
    const traceEmissions = {
        x: dates,
        y: df_nat.map(row => row.Total_Emissions_ktCO2),
        name: 'Total Emissions (ktCO2)',
        mode: 'lines',
        line: { color: '#ff5252', width: 2.5 }
    };
    const traceIntensity = {
        x: dates,
        y: df_nat.map(row => row.CO2_Intensity_gCO2_kWh * 100), // Scaled to display cleanly on the same graph
        name: 'CO2 Intensity (g/100kWh)',
        mode: 'lines',
        line: { color: '#3498db', width: 2.5, dash: 'dash' }
    };

    const layoutEm = {
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 50, r: 10, t: 10, b: 40 },
        xaxis: { title: 'Year', showgrid: false },
        yaxis: { title: 'Value', gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { orientation: 'h', y: 1.1, x: 1, xanchor: 'right' },
        hovermode: 'x unified'
    };
    Plotly.newPlot('chart-emissions-trend', [traceEmissions, traceIntensity], layoutEm, { responsive: true });
}

// Render Page 2: State Explorer
let stateChartPie = null;
let stateChartGen = null;
let rankChart = null;

function renderStateExplorer() {
    // 1. Populate Dropdown Select if empty
    const select = document.getElementById("state-selector");
    if (select.children.length <= 1) {
        // Extract unique states, ignoring totals
        const states = [...new Set(df_states.map(row => row.State))]
            .filter(st => st !== "India Total" && st !== "Others")
            .sort();
            
        states.forEach(state => {
            const opt = document.createElement("option");
            opt.value = state;
            opt.innerText = state;
            select.appendChild(opt);
        });

        // Event listener for dropdown
        select.value = "Gujarat"; // Default
        select.addEventListener("change", () => updateStateData(select.value));
        
        // Leaderboard metric selector listener
        document.getElementById("rank-metric-selector").addEventListener("change", renderLeaderboard);
    }

    updateStateData(select.value);
    renderLeaderboard();
}

function updateStateData(stateName) {
    const stateData = df_states.filter(row => row.State === stateName);
    if (!stateData.length) return;
    
    const latestStateRow = stateData[stateData.length - 1];

    // State Metric calculations
    const genCol = 'Electricity generation_Total Generation_GWh';
    const capSolarCol = 'Capacity_Solar_MW';
    const emCol = 'Power sector emissions_Total emissions_ktCO2';

    document.getElementById("state-metric-gen").innerHTML = `${formatNum(latestStateRow[genCol] || 0)} GWh`;
    document.getElementById("state-metric-solar").innerHTML = `${formatNum(latestStateRow[capSolarCol] || 0)} MW`;
    document.getElementById("state-metric-emissions").innerHTML = `${formatNum(latestStateRow[emCol] || 0, 1)} ktCO2`;

    // Capacity Mix Pie Chart
    const capKeys = {
        'Solar': 'Capacity_Solar_MW',
        'Wind': 'Capacity_Wind_MW',
        'Hydro': 'Capacity_Hydro_MW',
        'Coal': 'Capacity_Coal_MW',
        'Gas': 'Capacity_Gas_MW',
        'Nuclear': 'Capacity_Nuclear_MW',
        'Bioenergy': 'Capacity_Bioenergy_MW',
        'Other Fossil': 'Capacity_Other Fossil_MW'
    };

    const pieLabels = [];
    const pieValues = [];
    for (const [label, colKey] of Object.entries(capKeys)) {
        const val = latestStateRow[colKey] || 0;
        if (val > 0) {
            pieLabels.push(label);
            pieValues.push(val);
        }
    }

    const pieTrace = {
        labels: pieLabels,
        values: pieValues,
        type: 'pie',
        hole: 0.4,
        marker: {
            colors: ['#f1c40f', '#2ecc71', '#2980b9', '#2c3e50', '#e67e22', '#3498db', '#8e44ad', '#7f8c8d']
        },
        textinfo: 'percent+label',
        textposition: 'inside'
    };

    const pieLayout = {
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 20, r: 20, t: 20, b: 20 },
        showlegend: false
    };
    Plotly.newPlot('state-chart-pie', [pieTrace], pieLayout, { responsive: true });

    // State Generation Mix Area Chart
    const stateDates = stateData.map(row => row.Date);
    const genKeys = {
        'Coal': 'Electricity generation_Coal_GWh',
        'Gas': 'Electricity generation_Gas_GWh',
        'Hydro': 'Electricity generation_Hydro_GWh',
        'Nuclear': 'Electricity generation_Nuclear_GWh',
        'Solar': 'Electricity generation_Solar_GWh',
        'Wind': 'Electricity generation_Wind_GWh',
        'Bioenergy': 'Electricity generation_Bioenergy_GWh'
    };

    const fuelLabels = Object.keys(genKeys);
    const fuelColors = ['#2c3e50', '#e67e22', '#2980b9', '#3498db', '#f1c40f', '#2ecc71', '#8e44ad'];
    
    const genTraces = fuelLabels.map((fuel, fIdx) => {
        const colKey = genKeys[fuel];
        return {
            x: stateDates,
            y: stateData.map(row => row[colKey] || 0),
            name: fuel,
            stackgroup: 'one',
            fillcolor: fuelColors[fIdx],
            line: { color: fuelColors[fIdx], width: 1 }
        };
    });

    const genLayout = {
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 40, r: 10, t: 10, b: 40 },
        xaxis: { title: 'Date', showgrid: false },
        yaxis: { title: 'Generation (GWh)', gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { orientation: 'h', y: 1.1, x: 1, xanchor: 'right' },
        hovermode: 'x unified'
    };
    Plotly.newPlot('state-chart-gen', genTraces, genLayout, { responsive: true });
}

function renderLeaderboard() {
    const selectedMetric = document.getElementById("rank-metric-selector").value;
    
    // Map human names to columns
    const metricMap = {
        'solar_cap': 'Capacity_Solar_MW',
        'wind_cap': 'Capacity_Wind_MW',
        'fossil_cap': 'Capacity_Fossil_MW',
        'clean_gen_pct': 'Electricity generation_Clean_%',
        'total_emissions': 'Power sector emissions_Total emissions_ktCO2'
    };
    const metricYLabel = {
        'solar_cap': 'Solar Capacity (MW)',
        'wind_cap': 'Wind Capacity (MW)',
        'fossil_cap': 'Fossil Capacity (MW)',
        'clean_gen_pct': 'Clean Generation Share (%)',
        'total_emissions': 'Total Emissions (ktCO2)'
    };
    
    const targetCol = metricMap[selectedMetric];

    // Filter to latest date state-level data (exclude totals)
    const latestDateStr = df_states[df_states.length - 1].Date;
    const latestStatesData = df_states.filter(row => row.Date === latestDateStr && row.State !== "India Total" && row.State !== "Others");

    // Sort and take top 10
    const topStates = [...latestStatesData]
        .sort((a, b) => (b[targetCol] || 0) - (a[targetCol] || 0))
        .slice(0, 10);

    const xVals = topStates.map(row => row.State);
    const yVals = topStates.map(row => row[targetCol] || 0);

    const trace = {
        x: xVals,
        y: yVals,
        type: 'bar',
        marker: {
            color: yVals,
            colorscale: 'Viridis',
            showscale: false
        }
    };

    const layout = {
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 50, r: 10, t: 10, b: 60 },
        xaxis: { title: 'State', tickangle: -45, showgrid: false },
        yaxis: { title: metricYLabel[selectedMetric], gridcolor: 'rgba(255,255,255,0.05)' }
    };

    Plotly.newPlot('state-chart-leaderboard', [trace], layout, { responsive: true });
}

// Render Page 3: ML Forecast Sandbox
function renderMLForecastSimulator() {
    // Part 1: Autoregressive Ridge Demand Forecast
    calculateAndPlotForecast();

    // Part 2: Interactive Slider Scenario Setup
    const slider = document.getElementById("clean-share-slider");
    
    // Set slider initial value to current Clean Share
    const initCleanPct = Math.round(latestRow.Clean_Share * 100);
    slider.value = initCleanPct;
    document.getElementById("clean-share-val").innerText = `${initCleanPct}%`;

    slider.addEventListener("input", () => {
        const val = parseInt(slider.value);
        document.getElementById("clean-share-val").innerText = `${val}%`;
        simulateEmissionsScenario(val);
    });

    simulateEmissionsScenario(initCleanPct);
}

// 24-Month Demand Forecasting (Autoregressive Ridge Model in JS)
function calculateAndPlotForecast() {
    const histGen = df_nat.map(row => row.Total_Generation_GWh);
    const histDates = df_nat.map(row => row.Date); // Keep as raw string format (YYYY-MM-DD)
    
    const latestDateStr = histDates[histDates.length - 1];
    const parts = latestDateStr.split('-');
    let year = parseInt(parts[0], 10);
    let month = parseInt(parts[1], 10);
    
    const futureDates = [];
    const futurePreds = [];

    // Rolling autoregressive forecast step-by-step
    for (let i = 1; i <= 24; i++) {
        month++;
        if (month > 12) {
            month = 1;
            year++;
        }
        const dateStr = `${year}-${month.toString().padStart(2, '0')}-01`;
        futureDates.push(dateStr);

        const trend = df_nat.length + i - 1;

        // Lag calculations
        const lag1 = (futurePreds.length >= 1) 
            ? futurePreds[futurePreds.length - 1] 
            : histGen[histGen.length - 1];
            
        const lag2 = (futurePreds.length >= 2) 
            ? futurePreds[futurePreds.length - 2] 
            : histGen[histGen.length - 2];
            
        const lag12Idx = histGen.length + futurePreds.length - 12;
        const lag12 = (lag12Idx < histGen.length) 
            ? histGen[lag12Idx] 
            : futurePreds[lag12Idx - histGen.length];

        // Linear equation evaluation
        let pred = FORECAST_MODEL.intercept + FORECAST_MODEL.trend * trend;
        
        // Month categorical dummy
        if (month !== 1 && FORECAST_MODEL.monthCoefs[month]) {
            pred += FORECAST_MODEL.monthCoefs[month];
        }

        // Lag terms
        pred += FORECAST_MODEL.lag1 * lag1;
        pred += FORECAST_MODEL.lag2 * lag2;
        pred += FORECAST_MODEL.lag12 * lag12;

        futurePreds.push(pred);
    }

    // Chart forecast comparison
    const traceHist = {
        x: histDates,
        y: histGen,
        name: 'Historical Demand',
        mode: 'lines',
        line: { color: '#94a3b8', width: 2.5 } // Light slate gray for clean high-contrast readability
    };
    
    const tracePred = {
        x: futureDates,
        y: futurePreds,
        name: 'Forecasted Demand (Ridge)',
        mode: 'lines',
        line: { color: '#2ecc71', width: 3, dash: 'dash' }
    };

    const layout = {
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 45, r: 10, t: 10, b: 40 },
        xaxis: { title: 'Year', showgrid: false },
        yaxis: { title: 'Demand Requirement (GWh)', gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { orientation: 'h', y: 1.1, x: 1, xanchor: 'right' },
        hovermode: 'x unified'
    };

    Plotly.newPlot('forecast-chart', [traceHist, tracePred], layout, { responsive: true });
}

// Emissions Sandbox Simulator
function simulateEmissionsScenario(cleanPct) {
    const cleanFrac = cleanPct / 100;
    
    // Relative shares of clean fuels based on historical values
    const totCleanHist = latestRow.Solar_Share + latestRow.Wind_Share + latestRow.Hydro_Share + latestRow.Nuclear_Share;
    
    const solarRatio = latestRow.Solar_Share / totCleanHist;
    const windRatio = latestRow.Wind_Share / totCleanHist;
    const hydroRatio = latestRow.Hydro_Share / totCleanHist;
    const nuclearRatio = latestRow.Nuclear_Share / totCleanHist;

    const simSolar = cleanFrac * solarRatio;
    const simWind = cleanFrac * windRatio;
    const simHydro = cleanFrac * hydroRatio;
    const simNuc = cleanFrac * nuclearRatio;

    // Remaining fossil share
    const remainingFossil = 1.0 - cleanFrac;
    const totFossilHist = latestRow.Coal_Share + latestRow.Gas_Share;
    
    const coalRatio = latestRow.Coal_Share / totFossilHist;
    const gasRatio = latestRow.Gas_Share / totFossilHist;

    const simCoal = remainingFossil * coalRatio;
    const simGas = remainingFossil * gasRatio;

    // Build Table Mix percentages
    document.getElementById("sim-val-coal").innerText = `${(simCoal * 100).toFixed(1)}%`;
    document.getElementById("sim-val-gas").innerText = `${(simGas * 100).toFixed(1)}%`;
    document.getElementById("sim-val-solar").innerText = `${(simSolar * 100).toFixed(1)}%`;
    document.getElementById("sim-val-wind").innerText = `${(simWind * 100).toFixed(1)}%`;
    document.getElementById("sim-val-hydro").innerText = `${(simHydro * 100).toFixed(1)}%`;
    document.getElementById("sim-val-nuclear").innerText = `${(simNuc * 100).toFixed(1)}%`;

    // Run prediction formula
    const predictedIntensity = EMISSIONS_MODEL.intercept +
        EMISSIONS_MODEL.coalShare * simCoal +
        EMISSIONS_MODEL.gasShare * simGas +
        EMISSIONS_MODEL.solarShare * simSolar +
        EMISSIONS_MODEL.windShare * simWind +
        EMISSIONS_MODEL.hydroShare * simHydro +
        EMISSIONS_MODEL.nuclearShare * simNuc;

    // Display Gauge metrics
    document.getElementById("sim-metric-co2").innerText = `${predictedIntensity.toFixed(1)} gCO2/kWh`;
    
    const histIntensity = latestRow.CO2_Intensity_gCO2_kWh;
    const pctChange = ((predictedIntensity - histIntensity) / histIntensity) * 100;
    
    const deltaEl = document.getElementById("sim-delta-co2");
    deltaEl.innerText = `${pctChange >= 0 ? '+' : ''}${pctChange.toFixed(1)}% vs baseline`;
    deltaEl.className = `metric-delta ${predictedIntensity < histIntensity ? 'delta-up' : 'delta-down'}`;

    const statusEl = document.getElementById("sim-metric-status");
    statusEl.innerText = predictedIntensity < histIntensity ? "🟢 Decarbonizing" : "🔴 Fossil Heavy";

    // Gauge Chart Rendering
    const gaugeTrace = {
        type: "indicator",
        mode: "gauge+number",
        value: predictedIntensity,
        gauge: {
            axis: { range: [300, 700], tickwidth: 1, tickcolor: "white" },
            bar: { color: "#00d2ff" },
            bgcolor: "rgba(0,0,0,0)",
            borderwidth: 2,
            bordercolor: "rgba(255, 255, 255, 0.2)",
            steps: [
                { range: [300, 450], color: 'rgba(39, 174, 96, 0.4)' },
                { range: [450, 580], color: 'rgba(243, 156, 18, 0.4)' },
                { range: [580, 700], color: 'rgba(192, 57, 43, 0.4)' }
            ],
            threshold: {
                line: { color: "red", width: 4 },
                thickness: 0.75,
                value: histIntensity
            }
        }
    };

    const gaugeLayout = {
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 30, r: 30, t: 50, b: 10 },
        height: 250,
        font: { color: "white", family: "'Outfit', sans-serif" }
    };

    Plotly.newPlot('gauge-chart', [gaugeTrace], gaugeLayout, { responsive: true });
}
