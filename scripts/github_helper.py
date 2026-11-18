"""
PathSnap GitHub Workflow Helper
Handles issue listing, pull request creation, and merging via GitHub REST API.
"""
import os
import urllib.request
import json
import ssl
import sys
import certifi

REPO = "avulabalaji35-rgb/ml-hackathon"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not TOKEN and len(sys.argv) > 2 and sys.argv[-1].startswith("ghp_"):
    TOKEN = sys.argv[-1]

BASE_URL = f"https://api.github.com/repos/{REPO}"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "PathSnap-Automation"
}

ctx = ssl.create_default_context(cafile=certifi.where())

def api_call(endpoint, method="GET", data=None):
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        print(f"HTTP Error {e.code} on {method} {url}:\n{err_msg}", file=sys.stderr)
        raise

def list_open_issues():
    issues = api_call("issues?state=open")
    pure_issues = [i for i in issues if "pull_request" not in i]
    pure_issues.sort(key=lambda x: x["number"])
    print(f"Total open issues: {len(pure_issues)}")
    for i in pure_issues:
        print(f"  #{i['number']}: {i['title']}")
    return pure_issues

def create_pull_request(title, head, base="main", body=""):
    payload = {
        "title": title,
        "head": head,
        "base": base,
        "body": body
    }
    pr = api_call("pulls", method="POST", data=payload)
    print(f"Created PR #{pr['number']}: {pr['html_url']}")
    return pr

def merge_pull_request(pr_number, commit_title=None, commit_message=None):
    payload = {
        "merge_method": "merge"
    }
    if commit_title:
        payload["commit_title"] = commit_title
    if commit_message:
        payload["commit_message"] = commit_message
    res = api_call(f"pulls/{pr_number}/merge", method="PUT", data=payload)
    print(f"Merged PR #{pr_number}: {res.get('message', 'Success')}")
    return res

def close_issue(issue_number):
    res = api_call(f"issues/{issue_number}", method="PATCH", data={"state": "closed"})
    print(f"Closed issue #{issue_number}")
    return res

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "list"
    if action == "list":
        list_open_issues()
    elif action == "pr":
        title = sys.argv[2]
        head = sys.argv[3]
        body = sys.argv[4] if len(sys.argv) > 4 else ""
        create_pull_request(title, head, "main", body)
    elif action == "merge":
        pr_num = int(sys.argv[2])
        msg = sys.argv[3] if len(sys.argv) > 3 else "Merged pull request"
        merge_pull_request(pr_num, commit_message=msg)
    elif action == "close":
        issue_num = int(sys.argv[2])
        close_issue(issue_num)
