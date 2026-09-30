# Setup Guide — Reliable LLM on a new machine

This guide takes a fresh computer from `git clone` to a working app. It takes about 10–15 minutes, mostly waiting for downloads.

- **Windows** → follow [Part A](#part-a--windows-10--11)
- **Linux / macOS** → follow [Part B](#part-b--linux--macos)
- Both → then do [Configure a model](#configure-a-model-all-systems) and [Check it works](#check-it-works)

> **What is *not* in the repository (on purpose):** your API keys (`backend/.env`), the Python environment (`venv/`), frontend packages (`node_modules/`), and your uploaded documents and index (`backend/data/`). A new machine recreates these, and **your library starts empty**, so upload your documents again.

---

## Part A — Windows 10 / 11

### A1. Install the tools

Open **PowerShell** (Start menu → type *PowerShell*) and run:

```powershell
winget install -e --id Git.Git
winget install -e --id Python.Python.3.12
winget install -e --id OpenJS.NodeJS.LTS
```

Optional, for local models without an API key:

```powershell
winget install -e --id Ollama.Ollama
```

**Then close PowerShell and open a new one** so it picks up the new programs. Check them:

```powershell
git --version
py --version        # must be 3.10 or newer
node --version      # must be 18 or newer
```

> No `winget`? Install from the websites instead: git-scm.com, python.org (**tick "Add python.exe to PATH"**), nodejs.org (LTS), ollama.com.
>
> If `python` opens the Microsoft Store, turn off the alias: *Settings → Apps → Advanced app settings → App execution aliases* → switch off **python.exe** and **python3.exe**.

### A2. Clone the project

```powershell
cd $HOME\Documents
git clone https://github.com/ManojKumar-A-17/reliable_llm.git
cd reliable_llm
```

> Keep the path short and without special characters. `Documents\reliable_llm` is fine.

### A3. Run setup (one time)

Double-click **`setup.bat`** in the `reliable_llm` folder, or run:

```powershell
.\setup.bat
```

It creates `venv\`, installs the backend and frontend packages, creates `backend\.env`, and runs the tests. The first run takes several minutes.

> On Windows, pip installs the **CPU-only** PyTorch by default, so there's no extra download to trim.

### A4. Add a model

Open `backend\.env` in VS Code or Notepad and follow [Configure a model](#configure-a-model-all-systems) below.

### A5. Start the app

Double-click **`start.bat`**, or run:

```powershell
.\start.bat
```

- Two windows open, one for the backend and one for the frontend, each showing its logs.
- Your browser opens <http://localhost:3000> after a few seconds.
- To stop both servers, press **Enter** in the window where you ran `start.bat`.

If Windows Firewall asks about Python or Node, click **Allow**. It only listens on your own computer.

**Manual start** (two PowerShell windows), if you prefer:

```powershell
# Window 1 — backend
cd $HOME\Documents\reliable_llm\backend
..\venv\Scripts\activate
uvicorn api.server:app --reload --port 8000
```

```powershell
# Window 2 — frontend
cd $HOME\Documents\reliable_llm\frontend
npm run dev
```

> If `activate` says *running scripts is disabled*, run this once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Part B — Linux / macOS

### B1. Install the tools

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip curl
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

**macOS** (with [Homebrew](https://brew.sh))

```bash
brew install git python@3.12 node
```

Optional local models: install Ollama from <https://ollama.com/download>.

### B2. Clone, set up, start

```bash
git clone https://github.com/ManojKumar-A-17/reliable_llm.git
cd reliable_llm
bash setup.sh          # one time
# edit backend/.env — see "Configure a model"
bash start.sh          # Ctrl+C stops both servers
```

> **Linux without an NVIDIA GPU:** save about 3 GB with the CPU-only PyTorch build:
> ```bash
> source venv/bin/activate
> pip uninstall -y torch
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Configure a model (all systems)

Edit **`backend/.env`** and enable **at least one** provider. Restart the app after every change.

### Option A — Groq (recommended: free, fast, works on any laptop)

1. Create a free key at <https://console.groq.com/keys>
2. In `backend/.env`:
   ```
   LLM_PROVIDER_ORDER=groq,ollama,gemini
   GROQ_API_KEY=gsk_your_key_here
   ```

### Option B — Ollama (fully local, no key, but slow on CPU)

1. Download a model. On Windows, Ollama runs in the system tray after install.
   ```
   ollama pull llama3.2:3b
   ollama list
   ```
   The NAME shown by `ollama list` must match `OLLAMA_MODEL` exactly.
2. In `backend/.env`:
   ```
   LLM_PROVIDER_ORDER=ollama
   OLLAMA_ENABLED=true
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2:3b
   OLLAMA_FAST_MODEL=llama3.2:3b
   OLLAMA_NUM_CTX=4096
   OLLAMA_THINK=false
   TOP_K=4
   ```

### Option C — Google Gemini (free, but ~5 requests/minute)

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=<a model listed for your key in AI Studio>
```

**Best of both:** `LLM_PROVIDER_ORDER=groq,ollama` uses Groq for speed and switches to Ollama when Groq is rate-limited or you're offline.

> 🔒 **Keys go only in `backend/.env`** (git-ignored). Never put a real key in `.env.example` or any other tracked file, because GitHub will block the push.

---

## Check it works

1. Open <http://localhost:3000>. The top bar should say **Connected** and name your provider.
2. The first start downloads two small local models (~180 MB). Wait for `Local models ready` in the backend window.
3. Drag a PDF, DOCX, Markdown, HTML or TXT file into **Add documents**.
4. Ask a question the document answers, in **Fast** mode first.
5. You should see the answer stream in with citation numbers, a claim audit, and a confidence score.

---

## Day-to-day

| Task | Windows | Linux / macOS |
|---|---|---|
| Start the app | `.\start.bat` | `bash start.sh` |
| Get latest code | `git pull` then `.\setup.bat` if dependencies changed | `git pull` then `bash setup.sh` |
| Run tests | `cd backend` then `..\venv\Scripts\python -m pytest -q tests` | `cd backend && ../venv/bin/python -m pytest -q tests` |
| Save & push | `git add -A`, `git commit -m "msg"`, `git push` | same |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `py` / `python` not found, or it opens the Microsoft Store | Install Python 3.12 (A1), tick "Add to PATH", disable the Store aliases, open a new terminal. |
| `npm` not found | Install Node.js LTS (A1) and open a new terminal. |
| *running scripts is disabled on this system* | Use the `.bat` files, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |
| `pip install` fails building a package | Make sure you're on Python 3.10–3.12, then re-run setup. |
| `ModuleNotFoundError: No module named 'api'` | Start uvicorn from inside `backend`, or use `start.bat` / `start.sh`. |
| App shows **Backend offline** | Check the backend window for an error message. |
| **No LLM key configured** | Add a key (or enable Ollama) in `backend/.env`, then restart. |
| `model '…' not found` (Ollama) | `OLLAMA_MODEL` must match `ollama list` exactly, or run `ollama pull <model>`. |
| `Ollama is not running` | Start the Ollama app (Windows tray) or run `ollama serve`. |
| Answers take many minutes | Local models on CPU are slow. Use Groq, Fast mode, and `TOP_K=4`. |
| Model download fails (offline / firewall) | Set `RERANK_ENABLED=false` in `.env`. Search still works without the re-ranker. |
| Port 3000 or 8000 already in use | Close the old server windows. On Windows: `netstat -ano \| findstr :8000`, then `taskkill /PID <pid> /F`. |
| `.sh` scripts fail with `\r` errors (WSL / Git Bash) | Run `git config core.autocrlf input`, then re-clone. The repo's `.gitattributes` prevents this for new clones. |
| GitHub rejects push: "Push cannot contain secrets" | Remove the key from the file, rewrite the commit, and revoke the key. Never bypass the warning. |
