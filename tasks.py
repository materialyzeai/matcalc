"""Pyinvoke tasks.py file for automating releases and admin stuff."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pprint import pprint

import requests
from invoke import Context, task
from monty.os import cd

with open("pyproject.toml") as f:
    for line in f:
        if line.startswith("version"):
            NEW_VER = line.split("=")[-1].strip().strip('"')
            break


@task
def make_tutorials(ctx: Context):
    ctx.run("rm -rf docs/tutorials")
    ctx.run("jupyter nbconvert examples/*.ipynb --to=markdown --output-dir=docs/tutorials")
    for fn in glob.glob("docs/tutorials/*/*.png"):
        ctx.run(f'mv "{fn}" docs/assets')

    for fn in os.listdir("docs/tutorials"):
        lines = [
            "---",
            "layout: default",
            "title: " + fn,
            "nav_exclude: true",
            "---",
            "",
        ]
        path = f"docs/tutorials/{fn}"
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif fn.endswith(".md"):
            with open(path) as file:
                for line in file:
                    line = line.rstrip()  # noqa: PLW2901
                    if line.startswith("![png]"):
                        t1, t2 = line.split("(")
                        t2, t3 = t2.split("/")
                        lines.append(t1 + "(assets/" + t3)
                    else:
                        lines.append(line)
            with open(path, "w") as file:
                file.write("\n".join(lines))


@task
def make_docs(ctx: Context):
    """
    This new version requires markdown builder.

        pip install sphinx-markdown-builder

    Adding the following to conf.py

        extensions = [
            'sphinx_markdown_builder'
        ]

    Build markdown files with sphinx-build command

        sphinx-build -M markdown ./ build
    """
    make_tutorials(ctx)

    with cd("docs"):
        ctx.run("touch apidoc/index.rst", warn=True)
        ctx.run("rm matcalc.*.rst", warn=True)
        ctx.run("rm matcalc.*.html", warn=True)
        ctx.run("sphinx-apidoc --separate -P -M -d 3 -o apidoc -f ../src/matcalc")
        ctx.run("cp apidoc/modules.rst apidoc/index.rst")
        # Note: we use HTML building for the API docs to preserve search functionality.
        ctx.run("sphinx-build -b html apidoc html")  # HTML building.
        ctx.run("rm apidoc/*.rst", warn=True)
        ctx.run("mv html/matcalc*.html .")
        ctx.run("mv html/modules.html .")

        ctx.run("rm -r html", warn=True)
        ctx.run('sed -i "s/_static/assets/g" matcalc*.html')
        ctx.run("rm -rf doctrees", warn=True)

        ctx.run("cp ../README.md index.md")
        ctx.run("rm -rf *.orig doctrees", warn=True)

        with open("index.md") as f:
            contents = f.read()
        with open("index.md", "w") as f:
            contents = re.sub(
                r"\n## Official Documentation[^#]*",
                "{: .no_toc }\n\n## Table of contents\n{: .no_toc .text-delta }\n* TOC\n{:toc}\n\n",
                contents,
            )
            contents = "---\nlayout: default\ntitle: Home\nnav_order: 1\n---\n\n" + contents

            f.write(contents)


@task
def publish(ctx: Context):
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python -m build")
    ctx.run("python -m build --wheel")
    ctx.run("twine upload dist/*")


@task
def release(ctx: Context):  # noqa: ARG001
    desc = get_changelog()

    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "main",
        "name": "v" + NEW_VER,
        "body": desc,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialyzeai/matcalc/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]},
        timeout=10,
    )
    pprint(response.json())


def get_changelog():
    with open("changes.md") as f:
        contents = f.read()
        match = re.search(f"## v{NEW_VER}([^#]*)", contents)
        if not match:
            raise ValueError("could not parse latest version from changes.md")
        return match.group(1).strip()


def get_last_version():
    with open("changes.md") as f:
        contents = f.read()
        match = re.search("## v([^#]*)\n", contents)
        if not match:
            raise ValueError("could not parse latest version from changes.md")
        return match.group(1).split()[0].strip()
        return match.group(1).strip()


@task
def update_changelog(ctx: Context, version: str | None = None, *, dry_run: bool = False) -> None:
    """Create a preliminary change log using the git logs.

    Args:
        ctx (invoke.Context): The context object.
        version (str, optional): The version to use for the change log. If not provided, it will
            use the current date in the format 'YYYY.M.D'. Defaults to None.
        dry_run (bool, optional): If True, the function will only print the changes without
            updating the actual change log file. Defaults to False.
    """
    version = version or f"{datetime.now(tz=UTC):%Y.%-m.%-d}"
    lastver = get_last_version()
    print(f"Getting all comments since {lastver}")
    output = subprocess.check_output(["git", "log", "--pretty=format:%s", f"v{lastver}..HEAD"])
    lines = []
    ignored_commits = []
    for line in output.decode("utf-8").strip().split("\n"):
        re_match = re.match(r"Merge pull request \#(\d+)", line)
        if re_match and ("dependabot" not in line) and ("Pre-commit" not in line):
            pr_number = re_match[1].strip()
            print(f"Processing PR#{pr_number}")
            response = requests.get(
                f"https://api.github.com/repos/materialyzeai/matcalc/pulls/{pr_number}",
                timeout=60,
            )
            resp = response.json()
            lines += [f"- PR #{pr_number} {resp['title'].strip()} by @{resp['user']['login']}"]
            if body := resp["body"]:
                for ll in map(str.strip, body.split("\n")):
                    if ll in ("", "## Summary"):
                        continue
                    if ll.startswith(("## Checklist", "## TODO")):
                        break
                    lines += [f"    {ll}"]
        else:
            ignored_commits += [line]

    body = "\n".join(lines)
    try:
        # Use OpenAI to improve changelog. Requires openai to be installed and an OPENAPI_KEY env variable.
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAPI_KEY"])

        messages = [{"role": "user", "content": f"summarize as a markdown numbered list, include authors: '{body}'"}]
        chat = client.chat.completions.create(model="gpt-4o", messages=messages)

        reply = chat.choices[0].message.content
        body = "\n".join(reply.split("\n")[1:-1])
        body = body.strip().strip("`")
        print(f"ChatGPT Summary of Changes:\n{body}")

    except BaseException as ex:
        print(f"Unable to use openai due to {ex}")
    with open("changes.md", encoding="utf-8") as file:
        contents = file.read()
    delim = "##"
    tokens = contents.split(delim)
    tokens.insert(1, f"## v{version}\n\n{body}\n\n")
    if dry_run:
        print(tokens[0] + "##".join(tokens[1:]))
    else:
        with open("changes.md", mode="w", encoding="utf-8") as file:
            file.write(tokens[0] + "##".join(tokens[1:]))
        ctx.run("open changes.md")
    print("The following commit messages were not included...")
    print("\n".join(ignored_commits))


@task
def view_docs(ctx: Context) -> None:
    with cd("docs"):
        ctx.run("bundle exec jekyll serve")
