# Local notes for the Big Machine

## Git and SSH
This applies to all projects on this host (the "Big Machine").
- Git over SSH (fetch, pull, push, clone) uses keys from the user's SSH client via SSH agent forwarding.
  There are no usable private keys on this host.
- If `SSH_AUTH_SOCK` is not set or `ssh-add -l` fails in the shell session, look for the agent socket in `~/.ssh`
  (`ls ~/.ssh/*.sock`, e.g. `~/.ssh/agent-<short hostname>.sock`, a symlink to `/tmp/ssh-*/agent.*` created by the user's
  SSH session) and use it: `SSH_AUTH_SOCK=~/.ssh/agent-....sock git push ...`. Check it with `ssh-add -l`.
  - The socket target is in `/tmp`, so in a sandboxed shell with its own `/tmp` the agent is unreachable
    ("Error connecting to agent: No such file or directory"). Then run the Git command outside the sandbox.
  - Don't use other agent sockets in `/tmp`, and don't try the `-sk` key files directly.
  - If the socket is dead even outside the sandbox, the user has disconnected: ask them to reconnect their SSH session.
- The agent defaults to a key in the TPM of the SSH client, which needs no user interaction.
  It is somewhat buggy, and then SSH falls back to the user's YubiKeys (FIDO2 `-sk` keys), which need user interaction
  (passphrase and touch). So a Git SSH operation may wait for the user, or fail with an error such as
  "Please make sure you have the correct access rights" because of a mistyped passphrase or a missed touch,
  not because of the repository permissions or the network. Retry once before investigating further,
  and mention it to the user.
