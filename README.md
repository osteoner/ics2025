# 🧠 Community Detection via LLM Embeddings

This script performs **community detection** by leveraging **large language model (LLM) embeddings**. It extracts user content stored in a PostgreSQL database, and then uses clustering techniques (HDBSCAN) to identify hidden communities based on semantic similarity across user sequence representations.

---

## 📋 Command-Line Arguments

| Argument         | Default     | Description                              |
|------------------|-------------|------------------------------------------|
| `--db-host`      | `localhost` | Hostname of the database server.         |
| `--db-name`      | `cracked`   | Name of the underground forums database. |
| `--db-user`      | `""`        | Username for database authentication.    |
| `--db-password`  | `""`        | Password for the specified user.         |

---

## 🚀 Example Usage

```bash
python run.py --db-host 127.0.0.1 --db-name cracked --db-user admin --db-password secret
