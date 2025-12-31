(async function () {
  // --- Helper: Download function ---
  function downloadFile(content, fileName, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 0);
  }

  // --- Helper: Get Model Name for Filename ---
  function getModelFilename() {
    const modelEl = document.getElementById("task-model");
    let rawName = modelEl ? modelEl.textContent.trim() : "unknown_model";
    // Replace slashes or special chars with underscores to make it filesystem safe
    return rawName.replace(/[^a-zA-Z0-9.\-_]/g, "_");
  }

  const fileBaseName = getModelFilename();
  console.log(`📂 Target Filename Base: ${fileBaseName}`);

  // --- PART 1: Extract JSON (Instant) ---
  console.log("📥 Extracting JSON...");
  const jsonEl = document.getElementById("task-json-contents");
  if (jsonEl && jsonEl.textContent.trim()) {
    downloadFile(
      jsonEl.textContent,
      `info_${fileBaseName}.json`, // <--- UPDATED FILENAME
      "application/json"
    );
    console.log("✅ JSON downloaded.");
  } else {
    console.warn(
      "⚠️ JSON tab content not found. Make sure the JSON tab is loaded if you need the raw file."
    );
  }

  // --- PART 2: Scroll & Scrape Table (Index-Based) ---
  console.log("🚀 Starting Table Scrape (Index-Based)...");

  const scroller = document.querySelector('[data-testid="virtuoso-scroller"]');
  if (!scroller) {
    console.error(
      "❌ Could not find the table scroller. Ensure you are on the 'Samples' tab."
    );
    return;
  }

  // Storage: Key = Row Index (from DOM), Value = CSV String
  const collectedRows = new Map();
  const headers = ["Index", "ID", "Input", "Target", "Answer", "Score"];

  // Helper to clean text for CSV
  const escapeCsv = (txt) =>
    `"${(txt || "").replace(/"/g, '""').replace(/\n/g, " ")}"`;

  function scrapeVisibleRows() {
    const rowElements = document.querySelectorAll('[id^="sample-"]');

    rowElements.forEach((row) => {
      try {
        const wrapper = row.closest("[data-item-index]");
        const listIndex = wrapper
          ? wrapper.getAttribute("data-item-index")
          : null;

        if (listIndex === null || collectedRows.has(listIndex)) return;

        const idEl = row.querySelector(".sample-id");
        if (!idEl) return;

        const id = idEl.textContent || "";
        const input = row.querySelector(".sample-input")?.textContent || "";
        const target = row.querySelector(".sample-target")?.textContent || "";
        const answer = row.querySelector(".sample-answer")?.textContent || "";
        const score =
          row.querySelector('div[class*="_score_"]')?.textContent || "";

        collectedRows.set(
          listIndex,
          [
            listIndex,
            escapeCsv(id),
            escapeCsv(input),
            escapeCsv(target),
            escapeCsv(answer),
            escapeCsv(score),
          ].join(",")
        );
      } catch (e) {
        // Ignore errors
      }
    });
  }

  // --- Scroll Settings ---
  const scrollStep = 200;
  const delay = 0.75;
  let currentScroll = 0;

  // Reset to top
  scroller.scrollTop = 0;
  await new Promise((r) => setTimeout(r, 200));

  // Scroll Loop
  while (true) {
    scrapeVisibleRows();

    const maxScroll = scroller.scrollHeight - scroller.clientHeight;

    if (collectedRows.size % 200 === 0 && collectedRows.size > 0) {
      console.log(`Collecting... ${collectedRows.size} rows found.`);
    }

    if (Math.ceil(scroller.scrollTop) >= maxScroll) {
      break;
    }

    scroller.scrollTop += scrollStep;
    await new Promise((r) => setTimeout(r, delay));

    if (scroller.scrollTop === currentScroll) break;
    currentScroll = scroller.scrollTop;
  }

  // Final pass
  scrapeVisibleRows();

  // --- PART 3: Download ---
  if (collectedRows.size > 0) {
    const sortedRows = Array.from(collectedRows.entries())
      .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
      .map((entry) => entry[1]);

    const csvContent = headers.join(",") + "\n" + sortedRows.join("\n");

    downloadFile(csvContent, `${fileBaseName}.csv`, "text/csv;charset=utf-8;");
    console.log(`✅ Done! Sorted and collected ${sortedRows.length} rows.`);
  } else {
    console.error("❌ No rows collected.");
  }
})();
