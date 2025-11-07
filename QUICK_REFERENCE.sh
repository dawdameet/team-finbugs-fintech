#!/bin/bash
# Quick Reference: Reopen Failed Bug Issues
# ============================================

# MANUAL REOPEN (when you see verification failure in logs)

# Example log output:
# [✗] Team finbugs Bug #8 rejected (llm)
# [LOG] finbugs → Bug #8 (15 pts) - Verified: False

# Method 1: Using bash script
./reopen_issue.sh 8

# Method 2: Using Python script
python3 reopen_failed_bug.py 8

# Method 3: With custom reason
python3 reopen_failed_bug.py 8 --reason "LLM verification failed"

# AUTOMATED REOPEN (integrate into Flask webhook)
# Add this to your webhook handler:
<<'PYTHON_CODE'
from reopen_failed_bug import reopen_bug_issue

# Option A: Direct call with bug number
if not verified:
    success, msg = reopen_bug_issue(bug_number=8, reason="llm")
    print(msg)

# Option B: Auto-parse from log message
from webhook_integration import handle_bug_verification_failure

log_msg = "[✗] Team finbugs Bug #8 rejected (llm)"
handle_bug_verification_failure(log_msg)
PYTHON_CODE

# BULK REOPEN (multiple bugs at once)
for bug in 8 12 15; do
    ./reopen_issue.sh $bug
done

# CHECK ISSUE STATUS
gh issue view 29 -R dawdameet/team-finbugs-fintech

# LIST ALL OPEN BUG ISSUES
gh issue list -R dawdameet/team-finbugs-fintech --label domain-fintech --state open
