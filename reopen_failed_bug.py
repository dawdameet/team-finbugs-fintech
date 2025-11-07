#!/usr/bin/env python3
"""
Script to reopen GitHub issues when bug verification fails
Can be integrated with Flask webhook or run standalone

Usage:
    python3 reopen_failed_bug.py <bug_number>
    python3 reopen_failed_bug.py 8

Or import and use in webhook:
    from reopen_failed_bug import reopen_bug_issue, auto_reopen_on_failure
    
    # Method 1: Direct call
    reopen_bug_issue(bug_number=8, reason="LLM rejection")
    
    # Method 2: Auto-extract from log and reopen
    auto_reopen_on_failure(bug_id=8, verified=False, reason="llm")
"""

import subprocess
import sys
import json
import argparse
import re


def run_gh_command(args):
    """Run a GitHub CLI command and return the output"""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, e.stderr


def reopen_bug_issue(bug_number, repo="dawdameet/team-finbugs-fintech", reason=None):
    """
    Reopen a bug issue by bug number
    
    Args:
        bug_number: The bug ID (e.g., 8 for Bug #8)
        repo: GitHub repository in format "owner/repo"
        reason: Optional reason for reopening
    
    Returns:
        tuple: (success: bool, message: str)
    """
    print(f"[INFO] Searching for Bug #{bug_number}...")
    
    # Find the issue with the bug-X label
    stdout, stderr = run_gh_command([
        "gh", "issue", "list",
        "-R", repo,
        "--state", "all",
        "--label", f"bug-{bug_number}",
        "--json", "number,title,state",
        "--limit", "1"
    ])
    
    if stderr:
        return False, f"Error querying GitHub: {stderr}"
    
    try:
        issues = json.loads(stdout)
    except json.JSONDecodeError as e:
        return False, f"Error parsing GitHub response: {e}"
    
    if not issues:
        return False, f"Bug #{bug_number} not found in repository"
    
    issue = issues[0]
    issue_num = issue["number"]
    issue_title = issue["title"]
    issue_state = issue["state"]
    
    print(f"[INFO] Found Issue #{issue_num}: {issue_title}")
    print(f"[INFO] Current state: {issue_state}")
    
    if issue_state == "OPEN":
        return True, f"Issue #{issue_num} is already OPEN. No action needed."
    
    # Build comment
    comment = "Bug verification failed. Reopening for another attempt.\n\n"
    if reason:
        comment += f"**Verification Status:** Failed ({reason})\n"
    else:
        comment += "**Verification Status:** Failed\n"
    comment += "**Action:** Issue reopened automatically for retry"
    
    print(f"[INFO] Reopening issue #{issue_num}...")
    
    # Reopen the issue
    stdout, stderr = run_gh_command([
        "gh", "issue", "reopen",
        str(issue_num),
        "-R", repo,
        "--comment", comment
    ])
    
    if stderr:
        return False, f"Failed to reopen issue #{issue_num}: {stderr}"
    
    return True, f"Successfully reopened issue #{issue_num}"


def auto_reopen_on_failure(bug_id, verified, reason="llm", repo="dawdameet/team-finbugs-fintech"):
    """
    Automatically reopen issue if verification failed
    Use this directly in your Flask webhook after verification check
    
    Args:
        bug_id: The bug number (e.g., 8)
        verified: Boolean, verification result
        reason: Reason for failure (default: "llm")
        repo: GitHub repository
    
    Returns:
        bool: True if reopened or already open, False on error
    
    Example:
        verified = check_bug_fix(pr_data)
        auto_reopen_on_failure(bug_id=8, verified=verified, reason="llm")
    """
    if verified:
        # Bug was verified successfully, no need to reopen
        return True
    
    # Verification failed, reopen the issue
    print(f"[INFO] Bug #{bug_id} verification failed. Auto-reopening issue...")
    success, message = reopen_bug_issue(bug_number=bug_id, reason=reason, repo=repo)
    
    if success:
        print(f"[✓] Auto-reopen: {message}")
    else:
        print(f"[✗] Auto-reopen failed: {message}")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Reopen a GitHub issue when bug verification fails"
    )
    parser.add_argument(
        "bug_number",
        type=int,
        help="Bug number to reopen (e.g., 8 for Bug #8)"
    )
    parser.add_argument(
        "-r", "--repo",
        default="dawdameet/team-finbugs-fintech",
        help="GitHub repository (default: dawdameet/team-finbugs-fintech)"
    )
    parser.add_argument(
        "--reason",
        default="LLM rejection",
        help="Reason for reopening (default: LLM rejection)"
    )
    
    args = parser.parse_args()
    
    success, message = reopen_bug_issue(
        bug_number=args.bug_number,
        repo=args.repo,
        reason=args.reason
    )
    
    if success:
        print(f"[✓] {message}")
        return 0
    else:
        print(f"[✗] {message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
