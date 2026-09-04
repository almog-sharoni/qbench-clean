# GitHub Pages deployment

The repository includes a complete static documentation site and a dedicated
Pages workflow. Publishing requires a GitHub repository and approval to expose
its contents. No tokens or secrets belong in source files.

## Preview and validate locally

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
mkdocs build --strict
```

The generated `site/` directory is ignored by Git. Navigation, search, code-copy
buttons, light/dark themes, API/CLI references, and operational guides are built
from `mkdocs.yml` and `docs/`.

## First publication

After choosing the repository name and visibility:

1. Create the new GitHub repository and push this independent `main` branch.
2. In **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
3. Run the **Documentation** workflow or push a documentation change to `main`.
4. Wait for both the build and `github-pages` deployment jobs to succeed.
5. Open the deployment URL shown in the workflow environment.

The workflow uses GitHub's Pages configuration, artifact upload and deployment
actions, with deployment-only `pages: write` and `id-token: write` permissions.
Pull requests build docs but do not deploy. See GitHub's
[custom Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Site URL and custom domain

The default MkDocs configuration uses relative links, so the site works under a
project subpath. Once the repository is chosen, set `site_url` and `repo_url` in
`mkdocs.yml` to enable canonical URLs and a repository link. Configure custom domains
through GitHub Pages only when you control the relevant DNS.

GitHub Pages serves documentation, not the Streamlit application. Run the
[dashboard](dashboard.md) separately on an appropriate host.

## Release hygiene

Before the first public push, review `NOTICE.md`, choose a license if appropriate,
check the validation record, and ensure no private model artifacts were staged.
Pages deployment is a separate status from local build success: do not advertise
a live URL before GitHub reports a successful deployment.
