# Tripping Speed & W2W Calculation API

REST API built with Flask that accepts Excel files and returns tripping speed and weight-to-weight (W2W) calculations.

---

## Project Structure

```
.
├── server.py                   # Flask application and route definitions
├── calculations/
│   ├── tripping_speed.py       # Tripping speed algorithm
│   └── w2w.py                  # Weight-to-weight algorithm
├── requirements.txt
├── .env.example                # Environment variable template (copy to .env)
├── .gitignore
└── README.md
```

---

## Prerequisites

- Python **3.11** or higher
- `pip` (comes with Python)

---

## Setup

### 1. Git local setup

```bash
git config --local user.name "your name"
git config --local user.email "youremail@gitlab.sh"
```
### 2. Setup SSH key

- When asked for a file location, just press Enter to use the default
- When asked for a passphrase, you can press Enter to skip it

```bash
ssh-keygen -t ed25519 -C "your@gitlab.sh"
```

- Copy the entire output — it starts with ssh-ed25519 AAAA...
- Open your GitLab server in the browser
- Click your profile picture → Edit Profile
- Go to SSH Keys in the left sidebar
- Paste your key in the Key field
- Give it a title (any name like "My Laptop")
- Click Add Key

### 3. Clone the repository

```bash
git clone git@10.171.40.251:abelkhir/kpi-backend.git
cd <project-folder>
```

### 4. Create a virtual environment

A virtual environment isolates this project's dependencies from your system Python installation. Always use one.

```bash
python -m venv .venv
```

### 5. Activate the virtual environment

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

Once activated, your terminal prompt will be prefixed with `(.venv)`.

### 6. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Server

### Development mode

```bash
FLASK_DEBUG=true python server.py
```

The server will start on `http://localhost:5060`.


### Custom port

```bash
PORT=8080 python server.py
```

---

## Development Tips

### Use staging branch for development

Before you start coding swith to staging branch:
```bash
git checkout staging
```

Push implemented code to remote repo:
```bash
git push origin staging
```

Pull new changes:
```bash
git pull origin staging
```

### Adding a new dependency

```bash
pip install <package-name>
pip freeze > requirements.txt   # Update the requirements file
```

Never manually edit `requirements.txt` to add packages — always install first and then freeze.

### Deactivating the virtual environment

When you are done working on the project:
```bash
deactivate
```