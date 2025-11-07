#!/usr/bin/env python3
# raise_issues.py
# Creates GitHub issues for all 21 FinBugs from bugs.json
# Run with: python raise_issues.py
# Prerequisites: GitHub CLI (gh) authenticated; write access to "dawdameet/team-finbugs-fintech".

import subprocess
import json
import sys
import os

def create_labels(repo):
    """Create all necessary labels if they don't exist"""
    print(f"Creating labels for {repo}...")
    
    # Common labels
    labels = [
        {"name": "finbugs", "color": "0e8a16", "desc": "FinBugs Fintech module"},
        {"name": "puzzle", "color": "9f7ae2", "desc": "Puzzle-style bug with riddle hint"},
        {"name": "difficulty-easy", "color": "28a745", "desc": "Easy difficulty (10 pts)"},
        {"name": "difficulty-medium", "color": "ffc107", "desc": "Medium difficulty (20 pts)"},
        {"name": "difficulty-hard", "color": "fd7e14", "desc": "Hard difficulty (30 pts)"},
        {"name": "difficulty-extreme", "color": "dc3545", "desc": "Extreme difficulty (40 pts)"},
    ]
    
    # Bug-specific labels (bug-1 to bug-20, bug-32)
    bug_ids = list(range(1, 21)) + [32]
    for i in bug_ids:
        labels.append({
            "name": f"bug-{i}",
            "color": "d73a4a",  # Red for bugs
            "desc": f"Bug #{i} in FinBugs module"
        })
    
    for label in labels:
        try:
            subprocess.run([
                "gh", "label", "create", label["name"],
                "--repo", repo,
                "--color", label["color"],
                "--description", label["desc"]
            ], check=False, capture_output=True)
            print(f"✓ Created label: {label['name']}")
        except subprocess.CalledProcessError:
            print(f"  Label {label['name']} already exists; skipping.")

def create_bug_issues(repo, bugs_data):
    """Create bug issues with puzzle-style hints"""
    print(f"\nPopulating FinBugs issues for {repo}...")
    created_count = 0
    
    for bug in bugs_data:
        issue_body = f"""## 🐛 Bug Description
{bug['description']}

## ✅ Expected Behavior
{bug['expected']}

## ❌ Current Behavior
{bug['current']}

## 📁 Files Affected
`{bug['files']}`

## 🎯 Difficulty: {bug['difficulty'].upper()}
**Points: {bug['points']}**

## 🔍 Reproduction Steps
```
{bug['steps']}
```

## 💡 Hints
{bug.get('hints', 'No hints available - good luck!')}

## 🔧 Solution (Hidden)
<details>
<summary>Click to reveal solution</summary>

```
{bug.get('solution', 'Solution not provided')}
```
</details>

---
**Bug ID**: {bug['id']}  
**Module**: FinBugs (Fintech AI)  
**Status**: 🔴 UNRESOLVED
"""
        
        # Labels: bug-{id}, finbugs, difficulty-{level}, puzzle
        labels = [f"bug-{bug['id']}", "finbugs", f"difficulty-{bug['difficulty']}", "puzzle"]
        label_str = ",".join(labels)
        
        try:
            # Create issue using GitHub CLI
            result = subprocess.run([
                "gh", "issue", "create",
                "-R", repo,
                "-t", f"Bug {bug['id']}: {bug['title']}",
                "-b", issue_body,
                "--label", label_str
            ], check=True, capture_output=True, text=True)
            
            print(f"✓ Created issue for Bug {bug['id']}: {bug['title']}")
            created_count += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create issue for Bug {bug['id']}: {e}")
            if e.stderr:
                print(f"  Error: {e.stderr}")
    
    print(f"\n{'='*60}")
    print(f"Total issues created: {created_count}/{len(bugs_data)}")
    print(f"{'='*60}")

def load_bugs_from_json(json_path):
    """Load bugs from the bugs.json file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

# Target repo
repo = "dawdameet/team-finbugs-fintech"

if __name__ == "__main__":
    print("="*60)
    print("FinBugs GitHub Issues Creation Script")
    print("="*60)
    print(f"Target Repository: {repo}")
    
    # Check if gh CLI is installed
    try:
        subprocess.run(["gh", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n❌ Error: GitHub CLI (gh) is not installed or not in PATH")
        print("Install it from: https://cli.github.com/")
        sys.exit(1)
    
    # Check if authenticated
    try:
        subprocess.run(["gh", "auth", "status"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("\n❌ Error: Not authenticated with GitHub CLI")
        print("Run: gh auth login")
        sys.exit(1)
    
    # Load bugs from JSON file
    bugs_json_path = "/home/meet/College/CodeAI/codeverse/6-7-11/domains/fintech/bugs.json"
    
    # Check if file exists, if not try local path
    if not os.path.exists(bugs_json_path):
        bugs_json_path = "bugs.json"
    
    if not os.path.exists(bugs_json_path):
        print(f"\n❌ Error: bugs.json not found!")
        print(f"Expected at: /home/meet/College/CodeAI/codeverse/6-7-11/domains/fintech/bugs.json")
        print(f"or in current directory")
        sys.exit(1)
    
    print(f"Loading bugs from: {bugs_json_path}")
    finbugs_data = load_bugs_from_json(bugs_json_path)
    print(f"Total Bugs to Create: {len(finbugs_data)}")
    
    # Create labels first
    print("\n" + "="*60)
    print("STEP 1: Creating Labels")
    print("="*60)
    create_labels(repo)
    
    # Then create issues
    print("\n" + "="*60)
    print("STEP 2: Creating Issues")
    print("="*60)
    create_bug_issues(repo, finbugs_data)
    
    print("\n✅ All done! Check your GitHub Issues tab:")
    print(f"   https://github.com/{repo}/issues")
    print("\nTeams can now solve bugs and submit fixes!")
    print("Use 'Fixes #<issue-number>' in commit messages to auto-close issues.")
