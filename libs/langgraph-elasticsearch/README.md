# Langgraph Elasticsearch

Langgraph Elasticsearch is a library designed to integrate Langchain with Elasticsearch, providing powerful search capabilities for your applications.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install Langgraph Elasticsearch, you can use pip:

```bash
pip install langgraph-elasticsearch
```

Alternatively, you can clone the repository and install it manually:

```bash
git clone https://github.com/yourusername/langgraph-elasticsearch.git
cd langgraph-elasticsearch
pip install .
```

## Usage

Here is an example of how to use Langgraph Elasticsearch:

```python

from langgraph_elasticsearch import ElasticsearchMemoryStore

# Initialize the Elasticsearch client
es_client = ElasticsearchMemoryStore(
    es_url="http://localhost:9200",
    es_user="elastic",
    es_password="your_password"
)

# Index a document
doc = {
    "title": "Example Document",
    "content": "This is an example document for Langgraph Elasticsearch."
}
es_client.index_document(index="documents", id=1, document=doc)

# Search for a document
query = {
    "query": {
        "match": {
            "content": "example"
        }
    }
}
results = es_client.search(index="documents", body=query)
print(results)
```

## Contributing

We welcome contributions to Langgraph Elasticsearch! If you would like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.