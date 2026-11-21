## zsh
The default shell of most sessions on this computer is zsh, not bash, also in the shell tool of an agent.
Check with `echo $SHELL $ZSH_VERSION`. The differences from bash that have caused errors:
- An unquoted `$var` is not split into words: with `args="-C 0 text"`, `cmd $args` passes one argument `-C 0 text`.
  Use an array (`args=(-C 0 "two words")`, `cmd "${args[@]}"`), or split explicitly with `cmd ${=args}`.
  Arrays are indexed from 1.
- An unmatched glob is an error ("no matches found") instead of being passed on as is, also in arguments such as
  `grep -r --include=*.py`. Quote the patterns that are for the command (`--include='*.py'`), and use
  `setopt nullglob` before a loop over files that may not exist. Glob qualifiers such as `*(N)` may be disabled.
- For longer scripts written for bash, run them with `bash -c '...'` or put them in a file with a bash shebang.
