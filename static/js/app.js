
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) =>
    [...root.querySelectorAll(selector)];

const state = {
    busy: false,
    uploading: false,
    theme: localStorage.getItem("hr-assist-theme") || "system",
};

const elements = {
    chatForm: $("#chatForm"),
    question: $("#questionInput"),
    characterCount: $("#characterCount"),
    sendButton: $("#sendButton"),
    messages: $("#messagesList"),
    welcome: $("#welcomeScreen"),
    chatScroll: $("#chatScroll"),
    typing: $("#typingIndicator"),

    workflowStatus: $("#workflowStatus"),
    workflowList: $("#workflowList"),
    executionStatus: $("#executionStatus"),
    traceList: $("#traceList"),
    traceCount: $("#traceCount"),

    confidenceValue: $("#confidenceValue"),
    confidenceMeter: $("#confidenceMeter"),
    confidenceFill: $("#confidenceFill"),
    confidenceNote: $("#confidenceNote"),

    evidenceList: $("#evidenceList"),
    evidenceCount: $("#evidenceCount"),
    summarySource: $("#summarySource"),
    summaryMode: $("#summaryMode"),
    summaryGrade: $("#summaryGrade"),

    uploadModal: $("#uploadModal"),
    uploadForm: $("#uploadForm"),
    documentInput: $("#documentInput"),
    selectedFiles: $("#selectedFiles"),
    adminKey: $("#adminKeyInput"),
    uploadStatus: $("#uploadStatus"),
    submitUpload: $("#submitUploadBtn"),
    themeToggle: $("#themeToggleBtn"),
};

function scrollToBottom() {
    elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
}

function setStatus(element, text) {
    if (element) element.textContent = text;
}

function setBusy(busy) {
    state.busy = busy;
    elements.sendButton.disabled = busy;
    elements.question.disabled = busy;
    elements.typing.hidden = !busy;

    setStatus(
        elements.workflowStatus,
        busy ? "Working…" : "Ready"
    );
    setStatus(
        elements.executionStatus,
        busy ? "Running" : "Idle"
    );

    scrollToBottom();
}

function updateCharacterCount() {
    const length = elements.question.value.length;
    elements.characterCount.textContent = `${length} / 3000`;
}

function showWelcome() {
    elements.welcome.hidden = elements.messages.children.length > 0;
}

function appendUserMessage(question) {
    const template = $("#userMessageTemplate");
    const fragment = template.content.cloneNode(true);

    $(".user-message-content", fragment).textContent = question;

    elements.messages.appendChild(fragment);
    showWelcome();
    scrollToBottom();
}

function appendAssistantMessage(data) {
    const template = $("#assistantMessageTemplate");
    const fragment = template.content.cloneNode(true);

    const answerElement = $(".answer", fragment);
    const sourceChip = $(".source-chip", fragment);
    const citationRow = $(".citation-row", fragment);

    const answer = String(data.answer || "No answer was returned.");

    // Sanitize generated Markdown before inserting it into the page.
    if (window.marked && window.DOMPurify) {
        answerElement.innerHTML = DOMPurify.sanitize(
            marked.parse(answer)
        );
    } else {
        answerElement.textContent = answer;
    }

    const source = data.source_used || "unknown";
    sourceChip.textContent = `Source: ${source}`;
    sourceChip.hidden = false;

    renderCitations(citationRow, data.citations || []);

    elements.messages.appendChild(fragment);
    showWelcome();
    scrollToBottom();
}

function renderCitations(container, citations) {
    container.replaceChildren();

    if (!Array.isArray(citations) || citations.length === 0) {
        return;
    }

    for (const citation of citations) {
        const item =
            typeof citation === "string"
                ? { title: citation }
                : citation;

        if (!item || typeof item !== "object") continue;

        const title = String(
            item.title || item.source || item.name || "Source"
        );

        const url = item.url || item.source_url;

        if (typeof url === "string" && /^https?:\/\//i.test(url)) {
            const link = document.createElement("a");
            link.href = url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = title;
            container.appendChild(link);
        } else {
            const span = document.createElement("span");
            span.textContent = title;
            container.appendChild(span);
        }
    }
}

function renderTrace(trace) {
    elements.traceList.replaceChildren();

    if (!Array.isArray(trace) || trace.length === 0) {
        const empty = document.createElement("li");
        empty.className = "trace-empty";
        empty.textContent = "No execution trace was returned.";
        elements.traceList.appendChild(empty);
        setStatus(elements.traceCount, "0 steps");
        return;
    }

    for (const step of trace) {
        const item = document.createElement("li");
        const text = document.createElement("span");

        item.className = "trace-item";
        text.textContent =
            typeof step === "string"
                ? step
                : JSON.stringify(step);

        item.appendChild(text);
        elements.traceList.appendChild(item);
    }

    setStatus(elements.traceCount, `${trace.length} steps`);
}

function renderEvidence(citations) {
    elements.evidenceList.replaceChildren();

    if (!Array.isArray(citations) || citations.length === 0) {
        const empty = document.createElement("div");
        empty.className = "evidence-empty";
        empty.textContent = "No evidence sources were returned.";
        elements.evidenceList.appendChild(empty);
        setStatus(elements.evidenceCount, "0");
        return;
    }

    for (const citation of citations) {
        const item =
            typeof citation === "string"
                ? { title: citation }
                : citation;

        if (!item || typeof item !== "object") continue;

        const card = document.createElement("div");
        card.className = "evidence-item";

        const title = document.createElement("strong");
        title.textContent =
            item.title || item.source || item.name || "Evidence source";

        card.appendChild(title);

        if (item.snippet || item.content) {
            const description = document.createElement("p");
            description.textContent = item.snippet || item.content;
            card.appendChild(description);
        }

        const url = item.url || item.source_url;

        if (typeof url === "string" && /^https?:\/\//i.test(url)) {
            const link = document.createElement("a");
            link.href = url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = "Open source";
            card.appendChild(link);
        }

        elements.evidenceList.appendChild(card);
    }

    setStatus(
        elements.evidenceCount,
        String(elements.evidenceList.children.length)
    );
}

function updateObservability(data) {
    renderTrace(data.trace || []);
    renderEvidence(data.citations || []);

    setStatus(
        elements.summarySource,
        data.source_used || "unknown"
    );

    setStatus(
        elements.summaryMode,
        data.source_used === "web" ? "Web search" : "Automatic"
    );

    setStatus(
        elements.summaryGrade,
        data.evidence_grade || "Not provided"
    );

    const confidence = Number(data.confidence);

    if (Number.isFinite(confidence)) {
        // Accept either 0–1 or 0–100 confidence values.
        const percentage = Math.max(
            0,
            Math.min(100, confidence <= 1 ? confidence * 100 : confidence)
        );

        setStatus(
            elements.confidenceValue,
            `${Math.round(percentage)}%`
        );

        elements.confidenceFill.style.width = `${percentage}%`;
        elements.confidenceMeter.setAttribute(
            "aria-valuenow",
            String(Math.round(percentage))
        );

        setStatus(
            elements.confidenceNote,
            "Confidence supplied by the backend."
        );
    } else {
        setStatus(elements.confidenceValue, "—");
        elements.confidenceFill.style.width = "0%";
        elements.confidenceMeter.setAttribute("aria-valuenow", "0");

        setStatus(
            elements.confidenceNote,
            "The backend did not provide a confidence score."
        );
    }
}

async function readError(response) {
    try {
        const body = await response.json();
        return body.detail || body.message || `HTTP ${response.status}`;
    } catch {
        return `HTTP ${response.status}`;
    }
}

async function sendQuestion(question) {
    const cleaned = question.trim();

    if (cleaned.length < 2 || cleaned.length > 3000) {
        alert("Please enter a question between 2 and 3000 characters.");
        return;
    }

    if (state.busy) return;

    appendUserMessage(cleaned);
    elements.question.value = "";
    updateCharacterCount();
    setBusy(true);

    setStatus(elements.summarySource, "Processing");
    setStatus(elements.summaryGrade, "Pending");
    setStatus(elements.connectionStatus, "Contacting API");

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            body: JSON.stringify({ question: cleaned }),
        });

        if (!response.ok) {
            throw new Error(await readError(response));
        }

        const data = await response.json();

        appendAssistantMessage(data);
        updateObservability(data);
        setStatus(elements.connectionStatus, "Connected");
    } catch (error) {
        appendAssistantMessage({
            answer: `Sorry, I could not process your question. ${error.message}`,
            source_used: "error",
            citations: [],
            trace: [],
        });

        updateObservability({
            source_used: "error",
            trace: [],
            citations: [],
        });

        setStatus(elements.executionStatus, "Failed");
        setStatus(elements.connectionStatus, "Request failed");
    } finally {
        setBusy(false);
        elements.question.focus();
    }
}

// Chat form and suggested questions
elements.chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendQuestion(elements.question.value);
});

elements.question.addEventListener("input", updateCharacterCount);

elements.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        elements.chatForm.requestSubmit();
    }
});

$$(".suggestion-card").forEach((button) => {
    button.addEventListener("click", () => {
        const question = button.dataset.question;
        if (question) sendQuestion(question);
    });
});

// New conversation: clear visible messages and observability.
$("#newChatBtn").addEventListener("click", () => {
    if (state.busy) return;

    elements.messages.replaceChildren();
    elements.welcome.hidden = false;

    renderTrace([]);
    renderEvidence([]);

    setStatus(elements.executionStatus, "Idle");
    setStatus(elements.summarySource, "Not started");
    setStatus(elements.summaryMode, "Automatic");
    setStatus(elements.summaryGrade, "—");
    setStatus(elements.confidenceValue, "—");
    setStatus(elements.confidenceNote, "Confidence is shown when the backend provides a score.");

    elements.confidenceFill.style.width = "0%";
    elements.confidenceMeter.setAttribute("aria-valuenow", "0");
});

// Sidebar controls
$("#openSidebarBtn").addEventListener("click", () => {
    $("#sidebar").classList.add("is-open");
});

$("#closeSidebarBtn").addEventListener("click", () => {
    $("#sidebar").classList.remove("is-open");
});

// Clear the visible trace and evidence.
$("#clearTraceBtn").addEventListener("click", () => {
    renderTrace([]);
    renderEvidence([]);
});

// Theme toggle: system -> light -> dark -> system.
function applyTheme(theme) {
    state.theme = theme;

    if (theme === "system") {
        document.documentElement.removeAttribute("data-theme");
    } else {
        document.documentElement.dataset.theme = theme;
    }

    const labels = {
        system: "Theme: System — click to change",
        light: "Theme: Light — click to change",
        dark: "Theme: Dark — click to change",
    };

    elements.themeToggle.title = labels[theme];
    elements.themeToggle.setAttribute("aria-label", labels[theme]);
}

elements.themeToggle.addEventListener("click", () => {
    const next = {
        system: "light",
        light: "dark",
        dark: "system",
    };

    applyTheme(next[state.theme]);
    localStorage.setItem("hr-assist-theme", state.theme);
});

// Upload modal controls
function openUploadModal() {
    elements.uploadModal.hidden = false;
    elements.uploadStatus.textContent = "";
    elements.adminKey.focus();
}

function closeUploadModal() {
    if (state.uploading) return;
    elements.uploadModal.hidden = true;
}

$("#openUploadBtn").addEventListener("click", openUploadModal);
$("#uploadDocumentBtn").addEventListener("click", openUploadModal);
$("#closeUploadBtn").addEventListener("click", closeUploadModal);
$("#cancelUploadBtn").addEventListener("click", closeUploadModal);

elements.uploadModal.addEventListener("click", (event) => {
    if (event.target === elements.uploadModal) closeUploadModal();
});

elements.documentInput.addEventListener("change", () => {
    const files = [...elements.documentInput.files];

    elements.selectedFiles.textContent = files.length
        ? files.map((file) =>
            `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`
        ).join("\n")
        : "No files selected.";
});

elements.uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (state.uploading) return;

    const files = [...elements.documentInput.files];
    const adminKey = elements.adminKey.value;

    if (!files.length) {
        elements.uploadStatus.textContent = "Select at least one file.";
        return;
    }

    if (!adminKey.trim()) {
        elements.uploadStatus.textContent = "Enter your admin key.";
        return;
    }

    const allowedExtensions = [".pdf", ".txt", ".md", ".docx"];
    const invalidFile = files.find((file) => {
        const dot = file.name.lastIndexOf(".");
        const extension = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
        return !allowedExtensions.includes(extension);
    });

    if (invalidFile) {
        elements.uploadStatus.textContent =
            `Unsupported file type: ${invalidFile.name}`;
        return;
    }

    state.uploading = true;
    elements.submitUpload.disabled = true;
    elements.uploadStatus.textContent = "Uploading documents…";

    const results = [];

    try {
        // The backend endpoint accepts one file per request.
        for (const file of files) {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch("/api/ingest", {
                method: "POST",
                headers: {
                    "X-Admin-Key": adminKey,
                },
                body: formData,
            });

            if (!response.ok) {
                const message = await readError(response);
                results.push(`${file.name}: Failed — ${message}`);
                continue;
            }

            const result = await response.json();

            results.push(
                `${file.name}: Indexed successfully (${result.chunks} chunks)`
            );
        }

        elements.uploadStatus.textContent = results.join("\n");
    } catch (error) {
        elements.uploadStatus.textContent =
            `Upload request failed: ${error.message}`;
    } finally {
        state.uploading = false;
        elements.submitUpload.disabled = false;
    }
});

// Initial UI state
updateCharacterCount();
showWelcome();
applyTheme(state.theme);
renderTrace([]);
renderEvidence([]);
