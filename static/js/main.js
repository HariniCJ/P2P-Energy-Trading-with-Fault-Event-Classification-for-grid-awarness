// static/js/main.js

document.addEventListener("DOMContentLoaded", () => {
  const sampleSelector = document.getElementById("sampleSelector");
  const senderPeerSelector = document.getElementById("senderPeer");
  const receiverPeerSelector = document.getElementById("receiverPeer");
  const predictButton = document.getElementById("predictButton");
  const resultArea = document.getElementById("resultArea");
  const gridStatusText = document.getElementById("grid-status-text");
  const buttonText = document.getElementById("buttonText");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const plotDiv = document.getElementById("waveformPlot");

  const p2pGridComponents = {
    "prosumer-a": document.getElementById("prosumer-a"),
    "prosumer-b": document.getElementById("prosumer-b"),
    "community-battery": document.getElementById("community-battery"),
    "micro-industry": document.getElementById("micro-industry"),
    "microgrid-hub": document.getElementById("microgrid-hub"),
    "consumer-home": document.getElementById("consumer-home"),
    "ev-hub": document.getElementById("ev-hub"),
    "local-business": document.getElementById("local-business"),
    "consumer-load": document.getElementById("consumer-load"),
    "lines-p2p": document.getElementById("microgrid-hub"),
  };

  // Function to check if all required selections are made
  function checkSelections() {
    if (
      predictButton &&
      sampleSelector &&
      senderPeerSelector &&
      receiverPeerSelector
    ) {
      // Button is enabled if sampleSelector has a non-empty value.
      // senderPeer and receiverPeer always have a value by default.
      predictButton.disabled = !sampleSelector.value;
    }
  }

  function resetUI() {
    if (resultArea)
      resultArea.innerHTML =
        '<p class="text-gray-500 italic text-lg">Select peers, choose a grid event sample, and click "Analyze P2P Impact".</p>';
    if (gridStatusText) gridStatusText.textContent = "";
    if (plotDiv)
      plotDiv.innerHTML =
        '<p class="text-gray-500 italic text-center pt-10 text-lg">Select peers & event sample to view waveform.</p>';

    Object.values(p2pGridComponents).forEach((el) => {
      if (el)
        el.classList.remove("fault-indicator-red", "normal-indicator-green");
    });

    // Call checkSelections to correctly set button state
    checkSelections();

    if (loadingSpinner) loadingSpinner.classList.add("hidden");
    if (buttonText) buttonText.textContent = "Analyze P2P Impact";
  }

  function plotWaveform(waveform_data) {
    if (!plotDiv) return;

    if (
      !waveform_data ||
      !Array.isArray(waveform_data) ||
      waveform_data.length !== 3 ||
      !Array.isArray(waveform_data[0]) ||
      waveform_data[0].length === 0
    ) {
      plotDiv.innerHTML =
        '<p class="text-black italic text-center pt-10 text-lg">Error: Invalid waveform data for plotting.</p>';
      console.error(
        "Invalid waveform data received for plotting:",
        waveform_data
      );
      return;
    }
    const traceA = {
      y: waveform_data[0],
      mode: "lines",
      name: "Phase A",
      line: { color: "#000000", width: 1.5 },
    };
    const traceB = {
      y: waveform_data[1],
      mode: "lines",
      name: "Phase B",
      line: { color: "#555555", width: 1.5 },
    };
    const traceC = {
      y: waveform_data[2],
      mode: "lines",
      name: "Phase C",
      line: { color: "#999999", width: 1.5 },
    };
    const layout = {
      title: {
        text: "3-Phase Current Waveform (Grid Event)",
        font: { color: "#000000", size: 18 },
      },
      xaxis: {
        title: "Time Step (Sample Index)",
        color: "#000000",
        gridcolor: "#eeeeee",
        titlefont: { size: 14 },
        tickfont: { size: 12 },
      },
      yaxis: {
        title: "Current (Normalized)",
        color: "#000000",
        gridcolor: "#eeeeee",
        titlefont: { size: 14 },
        tickfont: { size: 12 },
      },
      margin: { l: 70, r: 40, b: 60, t: 60 },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: { color: "#000000", size: 14 },
      legend: {
        bgcolor: "rgba(255,255,255,0.7)",
        bordercolor: "#cccccc",
        font: { color: "#000000", size: 12 },
      },
      hovermode: "x unified",
    };
    Plotly.newPlot(plotDiv, [traceA, traceB, traceC], layout, {
      responsive: true,
    });
  }

  // Add event listeners only if the elements exist
  // Call checkSelections on change for all relevant dropdowns
  if (sampleSelector)
    sampleSelector.addEventListener("change", () => {
      resetUI(); // Reset UI parts but then re-check selections
      checkSelections();
    });
  if (senderPeerSelector)
    senderPeerSelector.addEventListener("change", () => {
      resetUI();
      checkSelections();
    });
  if (receiverPeerSelector)
    receiverPeerSelector.addEventListener("change", () => {
      resetUI();
      checkSelections();
    });

  if (predictButton) {
    predictButton.addEventListener("click", async () => {
      // Ensure selections are made before proceeding (redundant if button is correctly disabled, but safe)
      if (
        !sampleSelector.value ||
        !senderPeerSelector.value ||
        !receiverPeerSelector.value
      ) {
        alert(
          "Please ensure Sender Peer, Receiver Peer, and Grid Event Sample are all selected."
        );
        return;
      }

      const selectedSample = sampleSelector.value;
      const senderPeer = senderPeerSelector.value;
      const receiverPeer = receiverPeerSelector.value;

      predictButton.disabled = true;
      if (loadingSpinner) loadingSpinner.classList.remove("hidden");
      if (buttonText) buttonText.textContent = "Analyzing...";
      if (resultArea)
        resultArea.innerHTML =
          '<p class="text-black font-medium text-lg">Processing and analyzing P2P impact...</p>';
      if (plotDiv)
        plotDiv.innerHTML =
          '<p class="text-black italic text-center pt-10 text-lg">Loading waveform...</p>';
      if (gridStatusText) gridStatusText.textContent = "";

      Object.values(p2pGridComponents).forEach((el) => {
        if (el)
          el.classList.remove("fault-indicator-red", "normal-indicator-green");
      });

      let response;
      try {
        response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_filename: selectedSample,
            sender_peer: senderPeer,
            receiver_peer: receiverPeer,
          }),
        });

        if (!response.ok) {
          let errorData = {
            error: `Server responded with status: ${response.status} ${response.statusText}`,
          };
          try {
            const serverError = await response.json();
            if (serverError && serverError.error) {
              errorData = serverError;
            }
          } catch (e) {
            console.warn("Could not parse JSON error response from server:", e);
          }
          console.error("Server error response object:", response);
          console.error("Parsed/Default error data:", errorData);

          if (resultArea)
            resultArea.innerHTML = `<div class="border border-black p-5 rounded-lg text-lg" role="alert"><p class="font-bold text-black">Server Error</p><p class="text-black">${errorData.error}</p></div>`;
          if (gridStatusText)
            gridStatusText.textContent = "❌ Error during analysis";
          if (plotDiv)
            plotDiv.innerHTML =
              '<p class="text-black italic text-center pt-10 text-lg">Could not load waveform due to server error.</p>';
          return;
        }

        const result = await response.json();
        console.log("Received result from backend:", result);

        plotWaveform(result.waveform_data);

        const causesList =
          result.causes && Array.isArray(result.causes)
            ? result.causes
                .map((cause) => `<li class="ml-6 list-disc">${cause}</li>`)
                .join("")
            : "<li>Unknown or N/A</li>";

        const senderPeerName =
          typeof result.sender_peer === "string" && result.sender_peer
            ? result.sender_peer
                .replace(/-/g, " ")
                .replace(/\b\w/g, (l) => l.toUpperCase())
            : "N/A";
        const receiverPeerName =
          typeof result.receiver_peer === "string" && result.receiver_peer
            ? result.receiver_peer
                .replace(/-/g, " ")
                .replace(/\b\w/g, (l) => l.toUpperCase())
            : "N/A";

        const p2pTransactionHtml = `
                    <div class="mb-4 border-b border-gray-300 pb-4">
                        <h3 class="text-lg font-semibold text-black mb-2">P2P Transaction Context:</h3>
                        <p class="text-black text-base"><strong>Sender:</strong> ${senderPeerName}</p>
                        <p class="text-black text-base"><strong>Receiver:</strong> ${receiverPeerName}</p>
                    </div>
                `;

        if (resultArea)
          resultArea.innerHTML = `
                    ${p2pTransactionHtml}
                    <div class="border border-gray-300 p-5 rounded-lg">
                        <p class="text-base text-gray-600 mb-1"><strong>Grid Event Sample:</strong> ${
                          result.selected_sample || "N/A"
                        }</p>
                        <p class="text-xl font-semibold text-black">Detected Grid Event: ${
                          result.predicted_label || "N/A"
                        }</p>
                        <p class="text-base text-gray-600"><strong>Confidence:</strong> ${
                          result.confidence || "N/A"
                        }</p>
                        <p class="text-base text-gray-600"><strong>(Original Event Type:</strong> ${
                          result.true_label || "N/A"
                        })</p>
                    </div>
                    <div class="border-t border-gray-300 pt-5 mt-5">
                        <h3 class="text-lg font-semibold text-black mb-2">Event Description:</h3>
                        <p class="text-black text-base">${
                          result.description || "N/A"
                        }</p>
                    </div>
                    <div class="border-t border-gray-300 pt-5 mt-5">
                        <h3 class="text-lg font-semibold text-black mb-2">Potential Causes / Notes:</h3>
                        <ul class="text-black text-base space-y-1">
                            ${causesList}
                        </ul>
                    </div>
                    <div class="border-t border-gray-300 pt-5 mt-5">
                        <h3 class="text-lg font-semibold text-black mb-2">Impact on P2P Transfer:</h3>
                        <p class="text-black text-base">${
                          (result.p2p_impact_details &&
                            result.p2p_impact_details.p2p_disruption_message) ||
                          "Impact assessment pending."
                        }</p>
                    </div>
                `;

        Object.values(p2pGridComponents).forEach((el) => {
          if (el)
            el.classList.remove(
              "fault-indicator-red",
              "normal-indicator-green"
            );
        });

        const impactDetails = result.p2p_impact_details || {
          highlight_keys: ["microgrid-hub"],
          is_fault_condition: false,
          event_type: "Unknown",
        };
        const highlightKeys = impactDetails.highlight_keys || ["microgrid-hub"];
        const isFault = impactDetails.is_fault_condition;

        if (gridStatusText) {
          if (isFault) {
            gridStatusText.textContent = `⚠️ P2P Disrupted by Grid Fault (${
              impactDetails.event_type || "Fault"
            })`;
            gridStatusText.className =
              "text-center text-xl font-semibold mt-6 text-black break-words";
            let highlighted = false;
            highlightKeys.forEach((key) => {
              if (p2pGridComponents[key]) {
                p2pGridComponents[key].classList.add("fault-indicator-red");
                highlighted = true;
              }
            });
            if (!highlighted && p2pGridComponents["microgrid-hub"]) {
              p2pGridComponents["microgrid-hub"].classList.add(
                "fault-indicator-red"
              );
            }
          } else {
            gridStatusText.textContent = `✅ P2P Status: Stable / Monitoring Grid (${
              impactDetails.event_type || "Transient"
            })`;
            gridStatusText.className =
              "text-center text-xl font-semibold mt-6 text-black break-words";
            if (
              highlightKeys.includes("microgrid-hub") &&
              highlightKeys.length === 1
            ) {
              Object.values(p2pGridComponents).forEach((el) => {
                if (el) el.classList.add("normal-indicator-green");
              });
            } else {
              highlightKeys.forEach((key) => {
                if (p2pGridComponents[key]) {
                  p2pGridComponents[key].classList.add(
                    "normal-indicator-green"
                  );
                }
              });
            }
          }
        }
      } catch (error) {
        console.error(
          "Fetch API error OR JSON parsing error OR error processing successful response:",
          error
        );
        if (resultArea)
          resultArea.innerHTML = `<div class="border border-black p-5 rounded-lg text-lg" role="alert"><p class="font-bold text-black">Application Error</p><p class="text-black">Could not process the request or display results. Details: ${error.message}</p></div>`;
        if (gridStatusText) gridStatusText.textContent = "❌ Application Error";
        if (plotDiv)
          plotDiv.innerHTML =
            '<p class="text-black italic text-center pt-10 text-lg">Could not load waveform due to application error.</p>';
      } finally {
        // Call checkSelections to correctly set button state after prediction attempt
        if (predictButton) checkSelections();
        if (loadingSpinner) loadingSpinner.classList.add("hidden");
        if (buttonText) buttonText.textContent = "Analyze P2P Impact";
      }
    });
  }

  // Initial UI reset and button state check on page load
  resetUI();
});
