/* ═══════════════════════════════════════════════════════════════════════════
  Dharma Upadeshak — Frontend
   Chat  → POST /api/chat  (Gemini on server)
   STT   → Browser Web Speech API  (SpeechRecognition)
   TTS   → Browser Web Speech API  (SpeechSynthesis)
   ═══════════════════════════════════════════════════════════════════════════ */

const chatEl     = document.getElementById("chat");
const inputEl    = document.getElementById("msg");
const sendBtn    = document.getElementById("send-btn");
const micBtn     = document.getElementById("mic-btn");
const statusLine = document.getElementById("status-line");
const ttsToggle  = document.getElementById("tts-toggle");

let isLoading    = false;
let ttsEnabled   = true;
let isListening  = false;
let recognition  = null;
let chatLog      = [];

// ── Detect browser support ────────────────────────────────────────────────────
const hasSpeechRecognition = "SpeechRecognition" in window || "webkitSpeechRecognition" in window;
const hasSpeechSynthesis   = "speechSynthesis" in window;

if (!hasSpeechRecognition) micBtn.title = "Voice input not supported in this browser";
if (!hasSpeechSynthesis)   { ttsEnabled = false; ttsToggle.style.display = "none"; }

// ── Input helpers ─────────────────────────────────────────────────────────────
function onInput(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
  sendBtn.disabled = !el.value.trim() || isLoading;
}

function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function useSug(btn) {
  inputEl.value = btn.textContent.replace(/^"|"$/g, "");
  document.getElementById("suggestions")?.remove();
  sendBtn.disabled = false;
  sendMessage();
}

// ── Send chat ─────────────────────────────────────────────────────────────────
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isLoading) return;

  document.getElementById("suggestions")?.remove();
  appendBubble("user", text);
  chatLog.push({ role: "user", content: text, ts: new Date().toISOString() });

  inputEl.value = "";
  onInput(inputEl);
  setLoading(true);

  // Stop any ongoing TTS before sending
  if (hasSpeechSynthesis) speechSynthesis.cancel();

  try {
    const res  = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    appendBubble("assistant", data.reply);
    chatLog.push({ role: "assistant", content: data.reply, ts: new Date().toISOString() });

    if (ttsEnabled) speak(data.reply);

  } catch (err) {
    const msg = (err && err.message) ? err.message : "The connection was interrupted. Please try again.";
    appendBubble("assistant", msg);
  } finally {
    setLoading(false);
  }
}

// ── Bubble rendering ──────────────────────────────────────────────────────────
function appendBubble(role, text) {
  const msg    = mkEl("div", `msg ${role}`);
  const avatar = mkEl("div", "avatar");
  avatar.textContent = role === "user" ? "🙏" : "🪷";
  const bubble = mkEl("div", "bubble");
  bubble.innerHTML = renderText(text);
  msg.append(avatar, bubble);
  chatEl.appendChild(msg);
  chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
}

function renderText(t) {
  // Escape HTML first
  let s = t
    .replace(/&/g,"&amp;")
    .replace(/</g,"&lt;")
    .replace(/>/g,"&gt;");

  // Horizontal rule (---)
  s = s.replace(/^---$/gm, '<hr style="border-color:rgba(212,160,23,.2);margin:10px 0">');

  // ## Section headers  (## 🙏 Title)
  s = s.replace(/^## (.+)$/gm, (_, h) =>
    `<div class="section-header">${h}</div>`);

  // Bold  **text**
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Italic  *text*
  s = s.replace(/\*([^*\n]+?)\*/g, "<em>$1</em>");

  // Unordered list items  "- item" or "* item"
  s = s.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
  // Wrap consecutive <li> runs in <ul>
  s = s.replace(/(<li>.*<\/li>(\n|$))+/g, m => `<ul>${m}</ul>`);

  // Line breaks
  s = s.replace(/\n/g,"<br>");

  return s;
}

// ── Loading indicator ─────────────────────────────────────────────────────────
function setLoading(on) {
  isLoading = on;
  sendBtn.disabled = on;
  if (on) {
    const msg = mkEl("div", "msg assistant"); msg.id = "typing";
    msg.innerHTML = `<div class="avatar">🪷</div>
      <div class="bubble typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
    chatEl.appendChild(msg);
    chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
  } else {
    document.getElementById("typing")?.remove();
  }
}

// ── STT — Browser Web Speech API ─────────────────────────────────────────────
function toggleMic() {
  if (!hasSpeechRecognition) {
    showStatus("Voice input not supported. Try Chrome or Edge."); return;
  }

  if (isListening) {
    recognition?.stop(); return;
  }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.lang = "en-IN";           // Indian English; works for Hindi too
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.continuous = false;

  recognition.onstart = () => {
    isListening = true;
    micBtn.classList.add("recording");
    micBtn.textContent = "⏹️";
    showStatus("Listening… speak now");
    // Pause TTS while listening
    if (hasSpeechSynthesis) speechSynthesis.pause();
  };

  recognition.onresult = (e) => {
    const transcript = e.results[0][0].transcript.trim();
    if (transcript) {
      inputEl.value = transcript;
      onInput(inputEl);
      hideStatus();
      sendMessage();
    }
  };

  recognition.onerror = (e) => {
    showStatus(`Mic error: ${e.error}. Please try again.`);
    setTimeout(hideStatus, 3000);
  };

  recognition.onend = () => {
    isListening = false;
    micBtn.classList.remove("recording");
    micBtn.textContent = "🎙️";
    if (hasSpeechSynthesis) speechSynthesis.resume();
    hideStatus();
  };

  recognition.start();
}

// ── TTS — Browser Web Speech API ─────────────────────────────────────────────
function speak(text) {
  if (!hasSpeechSynthesis || !ttsEnabled) return;
  speechSynthesis.cancel();

  // Clean up for TTS: strip Devanagari, markdown, section headers, separators
  const clean = text
    .replace(/^---$/gm, "")                        // HR separators
    .replace(/^## .+$/gm, "")                      // section headers
    .replace(/[\u0900-\u097F\u1CD0-\u1CFF]+/g, "") // strip Devanagari
    .replace(/\*{1,2}(.+?)\*{1,2}/g, "$1")         // strip bold/italic
    .replace(/^[-•]\s+/gm, "")                      // strip list bullets
    .replace(/[🪷🙏✦⬇🙏💪🌿⭐🔥💡🌸🕉️]/g, "")      // strip emojis
    .trim();

  if (!clean) return;

  const utter   = new SpeechSynthesisUtterance(clean);
  utter.lang    = "en-IN";
  utter.rate    = 0.87;
  utter.pitch   = 1.0;
  utter.volume  = 1.0;

  // Pick a warm voice if available
  const voices  = speechSynthesis.getVoices();
  const pick    = voices.find(v => v.lang === "en-IN")
                || voices.find(v => v.lang.startsWith("en") && v.name.toLowerCase().includes("female"))
                || voices.find(v => v.lang.startsWith("en"));
  if (pick) utter.voice = pick;

  speechSynthesis.speak(utter);
}

// Voices load asynchronously in some browsers
if (hasSpeechSynthesis) {
  speechSynthesis.addEventListener("voiceschanged", () => {});
}

// ── TTS toggle ────────────────────────────────────────────────────────────────
function toggleTTS() {
  ttsEnabled = !ttsEnabled;
  if (!ttsEnabled) speechSynthesis.cancel();
  ttsToggle.textContent = ttsEnabled ? "🔊 Voice On" : "🔇 Voice Off";
}

// ── Save chat log ─────────────────────────────────────────────────────────────
function downloadLog() {
  if (!chatLog.length) { alert("No conversation yet."); return; }
  const ts   = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const blob = new Blob([JSON.stringify(chatLog, null, 2)], { type: "application/json" });
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = `dharma-upadeshak-${ts}.json`;
  a.click();
}

// ── Reset conversation ────────────────────────────────────────────────────────
async function resetChat() {
  if (!confirm("Start a new conversation?")) return;
  speechSynthesis?.cancel();
  await fetch("/api/reset", { method: "POST" });
  chatLog = [];
  chatEl.innerHTML = `
    <div class="msg assistant">
      <div class="avatar">🪷</div>
      <div class="bubble">Namaste, dear seeker. I am Dharma Upadeshak — your companion on the path of wisdom. Share what weighs upon your heart, and together we shall find the light of dharma.</div>
    </div>
    <div id="suggestions">
      <p>Begin with a question</p>
      <button class="sug" onclick="useSug(this)">"I feel lost and without purpose in life"</button>
      <button class="sug" onclick="useSug(this)">"How do I deal with betrayal from a close friend?"</button>
      <button class="sug" onclick="useSug(this)">"I'm facing financial hardship and feel hopeless"</button>
      <button class="sug" onclick="useSug(this)">"How do I stay calm in the face of injustice?"</button>
      <button class="sug" onclick="useSug(this)">"I have a difficult career decision to make"</button>
    </div>`;
}

// ── Utils ─────────────────────────────────────────────────────────────────────
function mkEl(tag, cls) { const e = document.createElement(tag); e.className = cls; return e; }
function showStatus(msg) { statusLine.textContent = msg; statusLine.style.display = "block"; }
function hideStatus()    { statusLine.style.display = "none"; }
