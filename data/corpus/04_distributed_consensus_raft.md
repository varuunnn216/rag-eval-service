# Distributed Consensus and Raft Protocol Reference

Distributed storage systems use the Raft consensus algorithm to manage a replicated state machine across a cluster of nodes while tolerating minority node failures.

## Cluster Quorum and Node Roles

Every Raft cluster requires an odd number of voting nodes $N$ (typically `3`, `5`, or `7` nodes). A quorum requires a strict majority of `floor(N / 2) + 1` nodes:
- A 3-node cluster requires a quorum of `2` nodes and tolerates `1` node failure.
- A 5-node cluster requires a quorum of `3` nodes and tolerates `2` node failures.
- A 7-node cluster requires a quorum of `4` nodes and tolerates `3` node failures.

At any given moment, each node is in one of three states: `Leader`, `Follower`, or `Candidate`. All client write requests are routed through the `Leader`.

## Leader Election and Timing Parameters

Time in Raft is divided into monotonically increasing logical integers called `terms`. Followers monitor leader liveness using a randomized election timeout:
- `election_timeout_ms`: Randomized uniformly between `150ms` and `300ms` per follower to prevent split votes during simultaneous elections.
- `heartbeat_interval_ms`: The active leader sends periodic empty `AppendEntries` RPC heartbeats every `30ms` (at least 5 times faster than the minimum election timeout) to suppress follower elections.

When a follower's election timer expires without receiving an `AppendEntries` RPC, it increments its `currentTerm`, transitions to `Candidate`, votes for itself, and broadcasts `RequestVote` RPCs in parallel.

## Log Replication Safety and Compaction

Raft guarantees the `Log Matching Property`: if two log entries in different nodes have the same index and the same term, then they store the exact same command and all preceding log entries are identical. Furthermore, the `Leader Completeness Property` ensures that a candidate cannot win an election unless its log is at least as up-to-date as a majority of the cluster (compared first by last log term, then by last log index).

To prevent unbounded disk growth, Raft periodically compacts committed log entries into a point-in-time snapshot once the log exceeds `snapshot_threshold_entries = 10000` entries or `snapshot_size_mb = 64`. Lagging followers whose required log entries have already been discarded receive state via the `InstallSnapshot` RPC.
