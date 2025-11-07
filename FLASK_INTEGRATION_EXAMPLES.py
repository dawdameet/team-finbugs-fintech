"""
FLASK WEBHOOK INTEGRATION - AUTO REOPEN ON VERIFICATION FAILURE
================================================================

Copy this code into your Flask webhook handler file.

This will automatically reopen GitHub issues when bug verification fails.
"""

# ============================================
# ADD THIS IMPORT AT THE TOP OF YOUR FLASK FILE
# ============================================
from reopen_failed_bug import auto_reopen_on_failure


# ============================================
# EXAMPLE 1: Simple Integration
# ============================================
@app.route('/webhook/github', methods=['POST'])
def github_webhook():
    """Your existing webhook handler"""
    
    data = request.json
    
    # Your existing code to extract bug_id and verify
    bug_id = extract_bug_id(data)  # Your function
    
    print(f"[INFO] Extracting code changes for Bug #{bug_id}...")
    
    # Your verification logic
    verified = verify_bug_fix(bug_id, data)  # Your function
    
    # Log result
    if verified:
        print(f"[✓] Team finbugs Bug #{bug_id} verified")
    else:
        print(f"[✗] Team finbugs Bug #{bug_id} rejected (llm)")
        print(f"[LOG] finbugs → Bug #{bug_id} (15 pts) - Verified: False")
    
    # ⭐ AUTO-REOPEN ON FAILURE - ADD THIS LINE ⭐
    auto_reopen_on_failure(bug_id=bug_id, verified=verified, reason="llm")
    
    return jsonify({"status": "ok"}), 200


# ============================================
# EXAMPLE 2: With Error Handling
# ============================================
@app.route('/webhook/github', methods=['POST'])
def github_webhook_safe():
    """Webhook with error handling"""
    
    try:
        data = request.json
        bug_id = extract_bug_id(data)
        
        print(f"[INFO] Extracting code changes for Bug #{bug_id}...")
        verified = verify_bug_fix(bug_id, data)
        
        if verified:
            print(f"[✓] Team finbugs Bug #{bug_id} verified")
        else:
            print(f"[✗] Team finbugs Bug #{bug_id} rejected (llm)")
            
            # Auto-reopen with error handling
            try:
                auto_reopen_on_failure(bug_id=bug_id, verified=verified, reason="llm")
            except Exception as e:
                print(f"[WARNING] Failed to auto-reopen Bug #{bug_id}: {e}")
        
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        print(f"[ERROR] Webhook processing failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================
# EXAMPLE 3: With Custom Repository
# ============================================
@app.route('/webhook/github', methods=['POST'])
def github_webhook_custom_repo():
    """Webhook with custom repository"""
    
    data = request.json
    bug_id = extract_bug_id(data)
    verified = verify_bug_fix(bug_id, data)
    
    if not verified:
        print(f"[✗] Team finbugs Bug #{bug_id} rejected (llm)")
        
        # Use with custom repo
        from reopen_failed_bug import reopen_bug_issue
        reopen_bug_issue(
            bug_number=bug_id,
            repo="dawdameet/team-finbugs-fintech",
            reason="llm"
        )
    
    return jsonify({"status": "ok"}), 200


# ============================================
# EXAMPLE 4: Inline Without Import
# ============================================
# If you don't want to import, copy this function into your Flask file:

import subprocess
import json

def quick_reopen_bug(bug_id, repo="dawdameet/team-finbugs-fintech"):
    """Quick inline function to reopen bug issue"""
    try:
        # Find issue with bug label
        result = subprocess.run(
            ["gh", "issue", "list", "-R", repo, "--state", "all", 
             "--label", f"bug-{bug_id}", "--json", "number,state", "--limit", "1"],
            capture_output=True, text=True, check=True
        )
        issues = json.loads(result.stdout)
        
        if not issues:
            return False
        
        issue = issues[0]
        if issue["state"] == "OPEN":
            print(f"[INFO] Bug #{bug_id} issue already OPEN")
            return True
        
        # Reopen issue
        subprocess.run(
            ["gh", "issue", "reopen", str(issue["number"]), "-R", repo,
             "--comment", "Bug verification failed. Reopening for retry.\n\n**Status:** Failed (LLM rejection)"],
            capture_output=True, text=True, check=True
        )
        print(f"[✓] Reopened Bug #{bug_id} issue #{issue['number']}")
        return True
        
    except Exception as e:
        print(f"[✗] Failed to reopen Bug #{bug_id}: {e}")
        return False

# Then use in webhook:
@app.route('/webhook/github', methods=['POST'])
def webhook_inline():
    data = request.json
    bug_id = extract_bug_id(data)
    verified = verify_bug_fix(bug_id, data)
    
    if not verified:
        print(f"[✗] Team finbugs Bug #{bug_id} rejected (llm)")
        quick_reopen_bug(bug_id)
    
    return jsonify({"status": "ok"}), 200


# ============================================
# INSTALLATION NOTES
# ============================================
"""
1. Make sure reopen_failed_bug.py is in the same directory as your Flask app
2. GitHub CLI (gh) must be installed and authenticated
3. Test with: python3 reopen_failed_bug.py 8

That's it! Issues will automatically reopen when verification fails.
"""
