# GitHub Issues Setup for FinBugs

This script automatically creates GitHub issues for all 21 bugs from `bugs.json`.

## Prerequisites

1. **GitHub CLI (gh)** must be installed and authenticated
   ```bash
   # Install gh (if not already installed)
   # On Ubuntu/Debian:
   sudo apt install gh
   
   # On macOS:
   brew install gh
   
   # Authenticate with GitHub
   gh auth login
   ```

2. **Repository Access**: You need write access to `dawdameet/team-finbugs-fintech`

## Usage

Simply run the script:

```bash
python raise_issues.py
```

or

```bash
python3 raise_issues.py
```

## What the Script Does

1. **Creates Labels** (if they don't exist):
   - `finbugs` - Main module label
   - `puzzle` - Marks it as a puzzle-style bug
   - `difficulty-easy`, `difficulty-medium`, `difficulty-hard`, `difficulty-extreme` - Difficulty levels
   - `bug-1` through `bug-20`, `bug-32` - Individual bug identifiers

2. **Creates Issues**: 
   - One issue per bug from `bugs.json`
   - Each issue includes:
     - Bug description
     - Expected vs Current behavior
     - Files affected
     - Difficulty level and points
     - Reproduction steps
     - Hints
     - Solution (hidden in spoiler)

## Issue Numbering

The script will create issues with GitHub's auto-incrementing issue numbers. The bug IDs are tracked in labels (`bug-1`, `bug-2`, etc.) and in the issue body.

## Verifying Issues

After running the script, visit:
```
https://github.com/dawdameet/team-finbugs-fintech/issues
```

## Troubleshooting

### Error: gh not found
```bash
# Install GitHub CLI first
sudo apt install gh  # Ubuntu/Debian
brew install gh      # macOS
```

### Error: Not authenticated
```bash
gh auth login
# Follow the prompts to authenticate
```

### Error: bugs.json not found
Make sure you're running the script from the repository root, or the bugs.json file is at:
```
/home/meet/College/CodeAI/codeverse/6-7-11/domains/fintech/bugs.json
```

## Output

The script will show progress like:
```
============================================================
FinBugs GitHub Issues Creation Script
============================================================
Target Repository: dawdameet/team-finbugs-fintech
Loading bugs from: /path/to/bugs.json
Total Bugs to Create: 21

============================================================
STEP 1: Creating Labels
============================================================
✓ Created label: finbugs
✓ Created label: difficulty-easy
...

============================================================
STEP 2: Creating Issues
============================================================
✓ Created issue for Bug 1: Incorrect VADER Sentiment Threshold
✓ Created issue for Bug 2: Reversed Stopword Filter Logic
...

============================================================
Total issues created: 21/21
============================================================

✅ All done! Check your GitHub Issues tab:
   https://github.com/dawdameet/team-finbugs-fintech/issues
```
