# Contributing to langchain-elasticsearch

If you have a bugfix or new feature that you would like to contribute to
langchain-elasticsearch, please find or open an issue about it first. Talk
about what you would like to do. It may be that somebody is already working on
it, or that there are particular issues that you should know about before
implementing the change.

We enjoy working with contributors to get their code accepted. There are many
approaches to fixing a problem and it is important to find the best approach
before writing too much code.

The LangChain contributing guide applies: https://docs.langchain.com/oss/python/contributing/overview

## Acceptable uses of LLMs

Generative AI can be a useful tool for contributors, but like any tool should be used with critical thinking and good judgement.

We encourage contributors to use AI tools efficiently where they help. However, AI assistance must be paired with meaningful human intervention, judgement, and contextual understanding. **If the human effort required to create a pull request is less than the effort required for maintainers to review it, that contribution should not be submitted.**

We struggle when contributors' entire work (code changes, documentation updates, pull request descriptions) are LLM-generated. These drive-by contributions often mean well but miss the mark in terms of contextual relevance, accuracy, and quality. Mass automated contributions like these represent a denial-of-service attack on our human effort.

**We will close pull requests and issues that appear to be low-effort, AI-generated spam.**

With great tools comes great responsibility.

## Setup Elasticsearch and langchain-elastic

Refer to the [main langchain-elastic docs](https://docs.langchain.com/oss/python/integrations/vectorstores/elasticsearch)
to see how to use langchain-elastic, including generating embeddings and
setting up Elasticsearch. Make sure to test your change manually.

## Contributing Code Changes

1. Run the linter and test suite to ensure your changes do not break existing code:

   ```
   # Go the package directory
   $ cd lib/elasticsearch

   # Install Poetry with Python 3.10
   $ uv virtualenv --python 3.10
   $ source .venv/bin/activate
   $ uv pip install poetry

   # Install dependencies
   $ make dev_install
   
   # Lint your changes
   $ make lint 
   
   # Run the test suite
   $ make test

   # Run integration tests
   $ export ES_URL=http://elastic:$ES_LOCAL_PASSWORD@localhost:9200
   $ curl $ES_URL  # check the $ES_URL is working
   $ make integration_test
   ```

2. Rebase your changes.
   Update your local repository with the most recent code from the main
   langchain-elastic repository, and rebase your branch on top of the latest `main`
   branch. We prefer your changes to be squashed into a single commit for easier
   backporting.

3. Submit a pull request. Push your local changes to your forked copy of the
   repository and submit a pull request. In the pull request, describe what your
   changes do and mention the number of the issue where discussion has taken
   place, eg “Closes #123″.  Please consider adding or modifying tests related to
   your changes.

Then sit back and wait. There will probably be a discussion about the pull
request and, if any changes are needed, we would love to work with you to get
your pull request merged into langchain-elastic.