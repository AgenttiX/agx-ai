-- owui2pdf pandoc filter
local known = {}
do
  local f = io.open(os.getenv("OWUI_LANGS") or "langs.txt", "r")
  if f then
    for line in f:lines() do
      line = line:gsub("%s+$", "")
      if line ~= "" then known[line:lower()] = true end
    end
    f:close()
  end
end

local alias = {
  py = "python", py3 = "python", python3 = "python", ipython = "python",
  sh = "bash", shell = "bash", console = "bash", shellsession = "bash",
  ["shell-session"] = "bash", fish = "bash", ksh = "bash",
  js = "javascript", mjs = "javascript", cjs = "javascript", node = "javascript",
  jsx = "javascriptreact", ts = "typescript", tsx = "typescript",
  yml = "yaml", jsonc = "json", json5 = "json", jsonl = "json",
  ["c++"] = "cpp", cxx = "cpp", cc = "cpp", hpp = "cpp", h = "c",
  csharp = "cs", ["c#"] = "cs", rs = "rust", rb = "ruby", kt = "kotlin",
  kts = "kotlin", golang = "go", docker = "dockerfile", md = "markdown",
  tex = "latex", ps1 = "powershell", pwsh = "powershell", bat = "dosbat",
  cmd = "dosbat", make = "makefile", mk = "makefile", objc = "objectivec",
  psql = "sqlpostgresql", postgres = "sqlpostgresql", postgresql = "sqlpostgresql",
  mysql = "sqlmysql", sqlite = "sql", vb = "monobasic", vbnet = "monobasic",
  hs = "haskell", ml = "ocaml", pl = "perl", jl = "julia", m = "matlab",
  cfg = "ini", conf = "ini", properties = "ini", env = "bash", dotenv = "bash",
  proto = "protobuf", tf = "terraform", hcl = "terraform", ex = "elixir",
  exs = "elixir", erl = "erlang", clj = "clojure", scm = "scheme", el = "commonlisp",
  lisp = "commonlisp", f90 = "fortranfree", f = "fortranfixed", fortran = "fortranfree",
  asm = "nasm", s = "gnuassembler", v = "verilog", sv = "systemverilog",
  htm = "html", xhtml = "html", svg = "xml", plist = "xml", xsl = "xslt",
  vue = "html", svelte = "html", scss = "scss", less = "css", styl = "css",
  gql = "graphql", cmake = "cmake", groovy = "groovy", gradle = "groovy",
  swift = "swift", dart = "dart", zig = "zig", nim = "nim", lua = "lua",
  r = "r", rmd = "markdown", sql = "sql", diff = "diff", patch = "diff",
  ["objective-c"] = "objectivec", ["objective-c++"] = "objectivecpp",
  ["c-sharp"] = "cs", plain = nil, text = nil, txt = nil, plaintext = nil,
}

local function plain_code_block(text)
  text = text:gsub("\\end{PlainCode}", "\\end {PlainCode}")
  return pandoc.RawBlock("latex",
    "\\begin{Shaded}\n\\begin{PlainCode}\n" .. text .. "\n\\end{PlainCode}\n\\end{Shaded}")
end

function CodeBlock(el)
  local lang = el.classes[1]
  if lang then
    lang = lang:lower():gsub("^language%-", "")
    lang = alias[lang] or lang
  end
  if lang and known[lang] then
    return pandoc.CodeBlock(el.text, pandoc.Attr("", {lang}, {}))
  end
  return plain_code_block(el.text)
end

-- Inline code: pandoc emits \texttt{...} with unbreakable spaces, so long
-- code spans overflow the line. Emit our own \texttt with break points.
local function tex_escape(s)
  s = s:gsub("[\\{}%$&#_%%~^]", function(c)
    if c == "\\" then return "\\textbackslash{}" end
    if c == "~" then return "\\textasciitilde{}" end
    if c == "^" then return "\\textasciicircum{}" end
    return "\\" .. c
  end)
  return s
end

function Code(el)
  local text = el.text
  if #text < 30 and not text:find("  ") then return nil end
  local out = {}
  local run = 0
  for _, cp in utf8.codes(text) do
    local ch = utf8.char(cp)
    if ch == " " then
      table.insert(out, "\\ \\allowbreak{}")
      run = 0
    else
      table.insert(out, tex_escape(ch))
      run = run + 1
      if ch:match("[/%.%-_:,;=%?&]") or run >= 20 then
        table.insert(out, "\\allowbreak{}")
        run = 0
      end
    end
  end
  return pandoc.RawInline("latex", "\\texttt{" .. table.concat(out) .. "}")
end

-- Emoji: wrap runs of emoji code points (including ZWJ sequences, variation
-- selectors, skin tones, flags) in \owuiemoji{} so HarfBuzz shapes them in
-- the emoji font as one unit.
local function is_emoji_base(cp)
  return (cp >= 0x1F000 and cp <= 0x1FAFF) or (cp >= 0x2600 and cp <= 0x27BF)
      or (cp >= 0x2300 and cp <= 0x23FF) or (cp >= 0x2B00 and cp <= 0x2BFF)
      or (cp >= 0x25AA and cp <= 0x25FE) or cp == 0x3030 or cp == 0x303D
      or cp == 0x3297 or cp == 0x3299
end
local function is_vs_only_base(cp) -- emoji only when followed by VS16/keycap
  return cp == 0xA9 or cp == 0xAE or cp == 0x2122 or cp == 0x2139
      or (cp >= 0x2194 and cp <= 0x2199) or cp == 0x21A9 or cp == 0x21AA
      or (cp >= 0x30 and cp <= 0x39) or cp == 0x23 or cp == 0x2A
end
local function is_modifier(cp)
  return cp == 0x200D or cp == 0xFE0F or cp == 0xFE0E or cp == 0x20E3
      or (cp >= 0x1F3FB and cp <= 0x1F3FF) or (cp >= 0xE0020 and cp <= 0xE007F)
end

local function emoji_raw(run)
  local esc = run:gsub("[#%%&$_{}]", "\\%0")
  return pandoc.RawInline("latex", "\\owuiemoji{" .. esc .. "}")
end

-- Very long words (no spaces) cannot be hyphenated; allow breaks every 25
-- characters so they do not run off the page.
local LONG_WORD = 40
local function soft_split(out, s)
  local n = utf8.len(s)
  if not n or n <= LONG_WORD then
    table.insert(out, pandoc.Str(s))
    return
  end
  local chunk, count = {}, 0
  for _, cp in utf8.codes(s) do
    table.insert(chunk, utf8.char(cp))
    count = count + 1
    if count == 25 then
      table.insert(out, pandoc.Str(table.concat(chunk)))
      table.insert(out, pandoc.RawInline("latex", "\\allowbreak{}"))
      chunk, count = {}, 0
    end
  end
  if #chunk > 0 then table.insert(out, pandoc.Str(table.concat(chunk))) end
end

-- Paragraphs containing such words additionally get \sloppy so that TeX
-- accepts the loose lines instead of producing an overfull line.
local function sloppy_wrap(el)
  -- Inline filters (Str) run before block filters, so long words have
  -- already been split into chunks joined by \allowbreak{}.
  for _, inl in ipairs(el.content) do
    if (inl.t == "RawInline" and inl.text:find("\\allowbreak{}", 1, true))
        or (inl.t == "Str" and (utf8.len(inl.text) or 0) > LONG_WORD) then
      el.content:insert(1, pandoc.RawInline("latex", "{\\sloppy{}"))
      el.content:insert(pandoc.RawInline("latex", "\\par}"))
      return el
    end
  end
end
function Para(el) return sloppy_wrap(el) end
function Plain(el) return sloppy_wrap(el) end

function Str(el)
  local s = el.text
  local quick = false
  for _, cp in utf8.codes(s) do
    if cp >= 0x2300 or cp == 0xA9 or cp == 0xAE then quick = true; break end
  end
  if not quick then
    if (utf8.len(s) or 0) > LONG_WORD then
      local out = {}
      soft_split(out, s)
      return out
    end
    return nil
  end

  local out, text, run = {}, {}, {}
  local prev_cp = nil
  local function flush_text()
    if #text > 0 then soft_split(out, table.concat(text)); text = {} end
  end
  local function flush_run()
    if #run > 0 then table.insert(out, emoji_raw(table.concat(run))); run = {} end
  end
  for _, cp in utf8.codes(s) do
    local ch = utf8.char(cp)
    if is_modifier(cp) then
      if #run > 0 then
        table.insert(run, ch)
      elseif prev_cp and is_vs_only_base(prev_cp) and #text > 0 then
        local base = table.remove(text)
        flush_text()
        run = {base, ch}
      else
        table.insert(text, ch)
      end
    elseif is_emoji_base(cp) then
      if #run == 0 then flush_text() end
      table.insert(run, ch)
    else
      if #run > 0 then
        if prev_cp == 0x200D then
          table.insert(run, ch) -- ZWJ joins a following non-emoji (rare)
        else
          flush_run()
          table.insert(text, ch)
        end
      else
        table.insert(text, ch)
      end
    end
    prev_cp = cp
  end
  flush_run()
  flush_text()
  if #out == 1 and out[1].t == "Str" then return nil end
  return out
end

-- Task lists ("- [x] item"): the commonmark reader leaves them as text when
-- extra extensions are enabled, so render the checkboxes ourselves.
function BulletList(el)
  local changed = false
  for _, item in ipairs(el.content) do
    local blk = item[1]
    if blk and (blk.t == "Plain" or blk.t == "Para") then
      local c = blk.content
      local first = c[1]
      if first and first.t == "Str" and (first.text == "[x]" or first.text == "[X]") then
        c[1] = pandoc.RawInline("latex", "$\\boxtimes$")
        changed = true
      elseif first and first.t == "Str" and first.text == "[" and c[2] and c[2].t == "Space"
          and c[3] and c[3].t == "Str" and c[3].text == "]" then
        c:remove(3); c:remove(2)
        c[1] = pandoc.RawInline("latex", "$\\square$")
        changed = true
      end
    end
  end
  if changed then return el end
end

-- Tables: the commonmark reader gives every column the default width, which
-- makes LaTeX typeset cells on one line and overflow the page. Assign
-- proportional widths when the content is wide.
function Table(el)
  local ncols = #el.colspecs
  if ncols == 0 then return nil end
  local maxlen, widest_row = {}, 0
  for i = 1, ncols do maxlen[i] = 1 end
  local function scan(rows)
    for _, row in ipairs(rows) do
      local rowlen = 0
      for i, cell in ipairs(row.cells) do
        if i <= ncols then
          local len = utf8.len(pandoc.utils.stringify(cell.contents)) or 0
          if len > maxlen[i] then maxlen[i] = len end
          rowlen = rowlen + len
        end
      end
      if rowlen > widest_row then widest_row = rowlen end
    end
  end
  scan(el.head.rows)
  for _, body in ipairs(el.bodies) do scan(body.head); scan(body.body) end
  scan(el.foot.rows)
  if widest_row + 3 * ncols <= 80 then return nil end
  local sum, w = 0, {}
  for i = 1, ncols do
    w[i] = math.max(math.min(maxlen[i], 60), 8)
    sum = sum + w[i]
  end
  local specs = {}
  for i = 1, ncols do
    specs[i] = {el.colspecs[i][1], w[i] / sum}
  end
  el.colspecs = specs
  return el
end

-- Images whose source is not a readable local file (e.g. /api/v1/files/...)
-- would abort LaTeX; replace them with a textual placeholder.
function Image(el)
  local src = el.src
  local f = io.open(src, "rb")
  if f then f:close(); return nil end
  local alt = pandoc.utils.stringify(el.caption)
  local label = (alt ~= "" and alt) or src
  return pandoc.Emph({pandoc.Str("[image: "), pandoc.Link({pandoc.Str(label)}, src), pandoc.Str("]")})
end
