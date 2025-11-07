"""
Flask Webhook Integration Example for Auto-Reopening Failed Bug Verifications

Add this to your Flask webhook handler to automatically reopen issues when verification fails.
"""

from reopen_failed_bug import reopen_bug_issue
import re


def handle_bug_verification_failure(log_message):
    """
    Parse verification failure log and reopen the issue
    
    Args:
        log_message: Log string like "[✗] Team finbugs Bug #8 rejected (llm)"
    
    Returns:
        bool: True if issue was reopened, False otherwise
    """
    # Extract bug number from log message
    match = re.search(r'Bug #(\d+)', log_message)
    if not match:
        print(f"[WARNING] Could not extract bug number from: {log_message}")
        return False
    
    bug_number = int(match.group(1))
    
    # Extract reason if available
    reason_match = re.search(r'rejected \(([^)]+)\)', log_message)
    reason = reason_match.group(1) if reason_match else "verification failed"
    
    print(f"[INFO] Verification failed for Bug #{bug_number}. Reopening issue...")
    
    success, message = reopen_bug_issue(bug_number=bug_number, reason=reason)
    
    if success:
        print(f"[✓] {message}")
        return True
    else:
        print(f"[✗] {message}")
        return False


# Example Flask webhook route integration
"""
@app.route('/webhook/github', methods=['POST'])
def github_webhook():
    # Your existing webhook code...
    
    # After verification check
    if not verified:
        log_msg = f"[✗] Team finbugs Bug #{bug_id} rejected (llm)"
        print(log_msg)
        
        # Auto-reopen the issue
        handle_bug_verification_failure(log_msg)
    
    return jsonify({"status": "ok"}), 200
"""


# Example standalone usage
if __name__ == "__main__":
    # Simulate a failed verification log
    test_log = "[✗] Team finbugs Bug #8 rejected (llm)"
    handle_bug_verification_failure(test_log)
