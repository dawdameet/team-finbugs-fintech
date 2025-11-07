# Auto-Reopen Failed Bug Issues

Scripts to automatically reopen GitHub issues when bug verification fails.

## Files

1. **`reopen_issue.sh`** - Bash script for manual/command-line usage
2. **`reopen_failed_bug.py`** - Python script with Flask integration support
3. **`webhook_integration.py`** - Example Flask webhook integration

## Usage

### Option 1: Bash Script (Manual)

```bash
# Reopen Bug #8
./reopen_issue.sh 8

# Reopen Bug #15
./reopen_issue.sh 15
```

### Option 2: Python Script (Manual)

```bash
# Basic usage
python3 reopen_failed_bug.py 8

# With custom reason
python3 reopen_failed_bug.py 8 --reason "LLM rejection"

# Different repository
python3 reopen_failed_bug.py 8 -r owner/repo --reason "Test failed"
```

### Option 3: Flask Webhook Integration

```python
from reopen_failed_bug import reopen_bug_issue

@app.route('/webhook/github', methods=['POST'])
def github_webhook():
    # ... your verification logic ...
    
    if not verified:
        bug_number = 8  # Extract from your webhook data
        success, message = reopen_bug_issue(
            bug_number=bug_number,
            reason="llm"
        )
        print(message)
    
    return jsonify({"status": "ok"}), 200
```

### Option 4: Parse Log and Auto-Reopen

```python
from webhook_integration import handle_bug_verification_failure

# Your verification log
log = "[✗] Team finbugs Bug #8 rejected (llm)"

# This will automatically extract bug number and reopen
handle_bug_verification_failure(log)
```

## How It Works

1. Searches for the issue with label `bug-X` (e.g., `bug-8`)
2. Checks if issue is already OPEN
3. If CLOSED, reopens it with a comment explaining the failure
4. Adds verification failure details to the comment

## Requirements

- GitHub CLI (`gh`) must be installed and authenticated
- Python 3.6+ (for Python scripts)
- Bash (for shell script)

## Install GitHub CLI

```bash
# Ubuntu/Debian
sudo apt install gh

# Authenticate
gh auth login
```

## Example Output

```
[INFO] Searching for Bug #8...
[INFO] Found Issue #29: Bug 8: Portfolio Weights Don't Sum to One
[INFO] Current state: CLOSED
[INFO] Reopening issue #29...
[✓] Successfully reopened issue #29
```

## Testing

Test the scripts without actually closing/reopening:

```bash
# Check if script can find the issue
python3 reopen_failed_bug.py 8
```

The script will tell you if the issue is already OPEN and skip reopening.
