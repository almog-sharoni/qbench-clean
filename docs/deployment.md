# GitHub Pages deployment

The repository includes a complete static documentation site and a dedicated
Pages workflow. Publishing requires a GitHub repository and approval to expose
its contents. No tokens or secrets belong in source files.

Documentation builds and link checks run without an enabled Pages site. Non-PR
builds upload a `github-pages` artifact containing the static site, downloadable
from the workflow run. Deployment is disabled unless the repository variable
`QBENCH_PAGES_ENABLED` is exactly `true`. A successful build with deployment skipped
does **not** mean a website has been published.

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

!!! warning "Private source does not mean a private website"
    GitHub Pages sites are generally publicly accessible even when their source
    repository is private. Private-repository Pages also requires a supported paid
    plan. Confirm the intended site visibility and account entitlement before
    enabling publication; do not change repository visibility to work around this.
    See GitHub's [Pages availability and visibility guidance](https://docs.github.com/en/pages/getting-started-with-github-pages/securing-your-github-pages-site-with-https).

1. Create the new GitHub repository and push this independent `main` branch.
2. In **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
3. In **Settings → Secrets and variables → Actions → Variables**, create repository
   variable `QBENCH_PAGES_ENABLED` with value `true`.
4. Run the **Documentation** workflow or push a documentation change to `main`.
5. Wait for both the build and `github-pages` deployment jobs to succeed.
6. Open the deployment URL shown in the workflow environment.

The workflow uses GitHub's Pages configuration, artifact upload and deployment
actions with Node.js 24 support, with deployment-only `pages: write` and
`id-token: write` permissions. The build job never queries or creates a Pages site;
the deployment job validates an already-enabled site and does not auto-enable it.
Pull requests build docs but do not deploy. Remove the variable or set it to `false`
to stop future deployments (this does not unpublish an existing site). See GitHub's
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
