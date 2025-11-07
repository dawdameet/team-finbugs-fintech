#!/bin/bash

# Script to reopen a GitHub issue when bug verification fails
# Usage: ./reopen_issue.sh <bug_number>
# Example: ./reopen_issue.sh 8

REPO="dawdameet/team-finbugs-fintech"

if [ -z "$1" ]; then
    echo "Error: Bug number required"
    echo "Usage: $0 <bug_number>"
    exit 1
fi

BUG_NUM=$1

echo "Searching for Bug #$BUG_NUM..."

# Find the issue number with the bug-X label
ISSUE_DATA=$(gh issue list -R $REPO --state all --label "bug-$BUG_NUM" --json number,title,state --limit 1 2>&1)

if [ $? -ne 0 ]; then
    echo "Error querying GitHub: $ISSUE_DATA"
    exit 1
fi

ISSUE_NUM=$(echo "$ISSUE_DATA" | grep -o '"number":[0-9]*' | head -1 | grep -o '[0-9]*')
ISSUE_STATE=$(echo "$ISSUE_DATA" | grep -o '"state":"[A-Z]*"' | head -1 | cut -d'"' -f4)
ISSUE_TITLE=$(echo "$ISSUE_DATA" | grep -o '"title":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -z "$ISSUE_NUM" ]; then
    echo "Error: Bug #$BUG_NUM not found in repository"
    exit 1
fi

echo "Found Issue #$ISSUE_NUM: $ISSUE_TITLE"
echo "Current state: $ISSUE_STATE"

if [ "$ISSUE_STATE" == "OPEN" ]; then
    echo "Issue is already OPEN. No action needed."
    exit 0
fi

echo "Reopening issue #$ISSUE_NUM..."

# Reopen the issue with a comment
gh issue reopen $ISSUE_NUM -R $REPO --comment "Bug verification failed. Reopening for another attempt.

**Verification Status:** Failed (LLM rejection)
**Action:** Issue reopened automatically for retry" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Successfully reopened issue #$ISSUE_NUM"
else
    echo "✗ Failed to reopen issue #$ISSUE_NUM"
    exit 1
fi
