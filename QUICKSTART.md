# 🚀 QUICK START: Auto-Reopen Failed Bugs in Flask

## ⚡ 2-Minute Setup

### Step 1: Import the function

At the top of your Flask webhook file, add:

```python
from reopen_failed_bug import auto_reopen_on_failure
```

### Step 2: Add one line after verification

In your webhook handler, after you verify the bug fix:

```python
auto_reopen_on_failure(bug_id, verified)
```

### ✅ DONE! 

That's it! Issues will automatically reopen when verification fails.

---

## 📝 Complete Example

```python
from flask import Flask, request, jsonify
from reopen_failed_bug import auto_reopen_on_failure  # ⭐ ADD THIS

@app.route('/webhook/github', methods=['POST'])
def github_webhook():
    data = request.json
    
    # Your existing code
    bug_id = extract_bug_id(data)
    verified = verify_bug_fix(bug_id, data)
    
    # Your existing logging
    if verified:
        print(f"[✓] Team finbugs Bug #{bug_id} verified")
    else:
        print(f"[✗] Team finbugs Bug #{bug_id} rejected (llm)")
    
    # ⭐ ADD THIS ONE LINE ⭐
    auto_reopen_on_failure(bug_id, verified)
    
    return jsonify({"status": "ok"}), 200
```

---

## 🎯 What Happens

**Before:**
```
[✗] Team finbugs Bug #8 rejected (llm)
[LOG] finbugs → Bug #8 (15 pts) - Verified: False
```
Issue stays CLOSED. Team can't retry.

**After:**
```
[✗] Team finbugs Bug #8 rejected (llm)
[LOG] finbugs → Bug #8 (15 pts) - Verified: False
[INFO] Bug #8 verification failed. Auto-reopening issue...
[✓] Auto-reopen: Successfully reopened issue #29
```
Issue automatically REOPENS with comment. Team can retry immediately!

---

## 🔧 Requirements

- `reopen_failed_bug.py` must be in same directory as Flask app
- GitHub CLI (`gh`) must be installed and authenticated

```bash
# Install gh
sudo apt install gh

# Authenticate
gh auth login
```

---

## 🧪 Test It

```bash
# Test the function works
python3 -c "from reopen_failed_bug import auto_reopen_on_failure; auto_reopen_on_failure(8, False)"
```

---

## 📚 More Examples

See these files for more options:
- `SIMPLEST_INTEGRATION.py` - Minimal code examples
- `FLASK_INTEGRATION_EXAMPLES.py` - Multiple integration patterns
- `REOPEN_ISSUES_README.md` - Full documentation

---

**That's all you need!** Just import and call `auto_reopen_on_failure(bug_id, verified)` 🎉
