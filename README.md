# 🛠️ Database Connection Script

This script connects to a PostgreSQL database using parameters passed via command-line arguments. It is flexible and configurable.

---

## 📋 Command-Line Arguments

| Argument         | Default     | Description                              |
|------------------|-------------|------------------------------------------|
| `--db-host`      | `localhost` | Hostname of the database server.         |
| `--db-name`      | `cracked`   | Name of the target database.             |
| `--db-user`      | `""`        | Username for database authentication.    |
| `--db-password`  | `""`        | Password for the specified user.         |

---

## 🚀 Example Usage

```bash
python script.py --db-host 127.0.0.1 --db-name cracked --db-user admin --db-password secret
