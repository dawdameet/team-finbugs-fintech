"""
SIMPLEST FLASK INTEGRATION - ONE LINE
======================================

Just add these 2 lines to your existing Flask webhook:
"""

# At the top of your Flask file (with other imports):
from reopen_failed_bug import auto_reopen_on_failure

# In your webhook handler, after verification:
@app.route('/webhook/github', methods=['POST'])
def github_webhook():
    # ... your existing code ...
    
    bug_id = 8  # Extract from your webhook data
    verified = False  # Your verification result
    
    # ⭐ JUST ADD THIS ONE LINE ⭐
    auto_reopen_on_failure(bug_id, verified)
    
    # ... rest of your code ...
    return jsonify({"status": "ok"}), 200


"""
COMPLETE EXAMPLE - Copy & Paste This Into Your Flask File
==========================================================
"""

from flask import Flask, request, jsonify
from reopen_failed_bug import auto_reopen_on_failure  # ⭐ ADD THIS

app = Flask(__name__)

@app.route('/webhook/github', methods=['POST'])
def github_webhook():
    data = request.json
    
    # Extract bug ID (adjust to match your logic)
    bug_id = data.get('bug_id', 8)  # Replace with your extraction logic
    
    print(f"[INFO] Extracting code changes for Bug #{bug_id}...")
    
    # Your verification logic
    verified = verify_bug_fix(bug_id, data)  # Replace with your function
    
    # Log the result
    if verified:
        print(f"[✓] Team finbugs Bug #{bug_id} verified")
    else:
        print(f"[✗] Team finbugs Bug #{bug_id} rejected (llm)")
        print(f"[LOG] finbugs → Bug #{bug_id} (15 pts) - Verified: False")
    
    # ⭐ AUTO-REOPEN IF FAILED - JUST THIS ONE LINE ⭐
    auto_reopen_on_failure(bug_id, verified)
    
    # Update leaderboard, etc.
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)


"""
That's it! Now when this happens:

    [✗] Team finbugs Bug #8 rejected (llm)
    [LOG] finbugs → Bug #{bug_id} (15 pts) - Verified: False

The issue will automatically reopen with a comment:
    "Bug verification failed. Reopening for another attempt.
    
    **Verification Status:** Failed (llm)
    **Action:** Issue reopened automatically for retry"
"""
