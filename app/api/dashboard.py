from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>OSP Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 16px;
                background: #f3f4f6;
                color: #111827;
            }

            h1 {
                margin-bottom: 8px;
            }

            p {
                margin-top: 0;
                margin-bottom: 16px;
            }

            .section {
                background: white;
                border: 1px solid #d1d5db;
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 16px;
            }

            .row {
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
            }

            .card {
                flex: 1 1 420px;
                min-width: 320px;
            }

            label {
                display: block;
                font-weight: bold;
                margin-bottom: 8px;
            }

            input[type="file"] {
                margin-bottom: 10px;
            }

            button {
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 14px;
                cursor: pointer;
                font-size: 14px;
            }

            button:hover {
                background: #1d4ed8;
            }

            pre {
                background: #111827;
                color: #22c55e;
                padding: 12px;
                border-radius: 8px;
                overflow: auto;
                max-height: 220px;
                white-space: pre-wrap;
                word-break: break-word;
                margin-top: 12px;
            }

            iframe {
                width: 100%;
                height: 700px;
                border: 1px solid #d1d5db;
                border-radius: 10px;
                background: white;
            }

            .small-note {
                color: #4b5563;
                font-size: 13px;
                margin-top: 8px;
            }
        </style>
    </head>
    <body>
        <h1>OSP Redlining Dashboard</h1>
        <p>Upload route + bore CSV, then refresh the map and load the report.</p>

        <div class="row">
            <div class="section card">
                <h2>1. Upload Route File</h2>
                <label for="routeFile">KMZ / KML file</label>
                <input id="routeFile" type="file" />
                <br>
                <button onclick="uploadRoute()">Upload Route</button>
                <div class="small-note">Uploads the route and regenerates latest_route.geojson.</div>
                <pre id="routeResult">Waiting...</pre>
            </div>

            <div class="section card">
                <h2>2. Upload Bore CSV</h2>
                <label for="csvFile">CSV file</label>
                <input id="csvFile" type="file" />
                <br>
                <button onclick="uploadCsv()">Upload CSV</button>
                <div class="small-note">Expected columns: station, depth, boc</div>
                <pre id="csvResult">Waiting...</pre>
            </div>
        </div>

        <div class="section">
            <h2>3. Job Report</h2>
            <button onclick="loadReport()">Load Report</button>
            <pre id="reportResult">No report yet</pre>
        </div>

        <div class="section">
            <h2>4. Map Preview</h2>
            <button onclick="refreshMap()">Refresh Map</button>
            <div class="small-note">Performance mode: route + station markers only.</div>
            <iframe id="mapFrame" src="/api/map-preview"></iframe>
        </div>

        <script>
            async function uploadRoute() {
                const fileInput = document.getElementById("routeFile");
                const resultBox = document.getElementById("routeResult");

                if (!fileInput.files.length) {
                    resultBox.textContent = "Please choose a route file first.";
                    return;
                }

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                resultBox.textContent = "Uploading route...";

                try {
                    const response = await fetch("/api/upload", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    resultBox.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    resultBox.textContent = "Route upload failed: " + error;
                }
            }

            async function uploadCsv() {
                const fileInput = document.getElementById("csvFile");
                const resultBox = document.getElementById("csvResult");

                if (!fileInput.files.length) {
                    resultBox.textContent = "Please choose a CSV file first.";
                    return;
                }

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                formData.append("replace_existing", "true");

                resultBox.textContent = "Uploading CSV...";

                try {
                    const response = await fetch("/api/upload-bore-csv", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    resultBox.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    resultBox.textContent = "CSV upload failed: " + error;
                }
            }

            async function loadReport() {
                const resultBox = document.getElementById("reportResult");
                resultBox.textContent = "Loading report...";

                try {
                    const response = await fetch("/api/report-summary");
                    const data = await response.json();
                    resultBox.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    resultBox.textContent = "Report load failed: " + error;
                }
            }

            function refreshMap() {
                const frame = document.getElementById("mapFrame");
                frame.src = "/api/map-preview?ts=" + Date.now();
            }
        </script>
    </body>
    </html>
    """