# PostgreSQL MVCC, Transaction Isolation, and Vacuum Maintenance

PostgreSQL implements Multi-Version Concurrency Control (MVCC) so that readers never block writers and writers never block readers during concurrent database operations.

## Tuple Visibility and Transaction Isolation Levels

When a row is updated or deleted in PostgreSQL, the existing physical tuple is not overwritten in place. Instead, PostgreSQL marks the old tuple with the deleting transaction ID in its `xmax` system header and inserts a new tuple version with the creating transaction ID in `xmin`.

PostgreSQL supports three distinct transaction isolation levels (note that `Read Uncommitted` behaves identically to `Read Committed` in PostgreSQL):
- `Read Committed` (default): Each statement within a transaction sees a snapshot of data committed before that individual statement began.
- `Repeatable Read`: All statements in the transaction see a static snapshot taken at the start of the first non-control statement. Concurrent modifications to the same row abort the transaction with serialization error `SQLSTATE 40001`.
- `Serializable`: Uses Serializable Snapshot Isolation (SSI) with `SIReadLock` predicate locks to detect read-write dependency cycles (`rw-antidependencies`), guaranteeing serial equivalence or raising `SQLSTATE 40001`.

## HOT Updates, Autovacuum, and XID Wraparound Prevention

Heap-Only Tuple (HOT) updates occur when an `UPDATE` does not modify any column referenced by an index and sufficient free space exists on the same `8KB` heap page. Setting table `fillfactor = 80` reserves 20% of each page for future HOT updates, avoiding index bloat.

Because PostgreSQL uses 32-bit transaction counters, roughly 2 billion future transactions exist at any point. To prevent catastrophic transaction ID wraparound, autovacuum triggers an aggressive anti-wraparound freeze when a table's oldest unfrozen `xmin` exceeds `autovacuum_freeze_max_age = 200000000` (200 million transactions), replacing old `xmin` values with `FrozenTransactionId`.

## Write-Ahead Logging (WAL) and Checkpoints

Durability is enforced by writing all modifications to the Write-Ahead Log (WAL) before dirty heap pages are flushed to disk. For streaming replication and point-in-time recovery (PITR), `wal_level` must be set to `replica` or `logical`. Production instances tune `max_wal_size = 4GB` and `checkpoint_timeout = 15min` with `checkpoint_completion_target = 0.9` to spread checkpoint I/O evenly.
