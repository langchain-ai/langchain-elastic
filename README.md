# 🦜️🔗 LangChain Elastic

This repository contains 1 package with ElasticSearch integrations with LangChain:

- [langchain-elasticsearch](https://pypi.org/project/langchain-elasticsearch/) integrates [ElasticSearch](https://www.elastic.co/elasticsearch).

## Initial Repo Checklist (Remove this section after completing)

This setup assumes that the partner package is already split. For those instructions,
see [these docs](https://python.langchain.com/docs/contributing/integrations#partner-packages).

- [x] Fill out the readme above (for folks that follow pypi link)
- [x] Copy package into /libs folder
- [x] Update these fields in /libs/*/pyproject.toml

    - `tool.poetry.repository`
    - `tool.poetry.urls["Source Code"]`
    
- [x] Add integration testing secrets in Github (ask Erick for help)
- [x] Add secrets as env vars in .github/workflows/_release.yml
- [x] Configure `LIB_DIRS` in .github/scripts/check_diff.py
- [x] Add partner collaborators in Github (ask Erick for help)
- [x] Add new repo to test-pypi and pypi trusted publishing (ask Erick for help)
- [x] Populate .github/workflows/_release.yml with `on.workflow_dispatch.inputs.working-directory.default`
