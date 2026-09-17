-- Pandoc reads the documents' empty HTML anchors as inline tags, and drops
-- them when generating LaTeX. Keep both inline and block forms as explicit
-- destinations so the Markdown contents lists and section links work in PDF.
function RawInline(el)
  if FORMAT:match('latex') and el.format == 'html' then
    local id = el.text:match('^<a id="([%w_:.-]+)">$')
    if id then
      return pandoc.RawInline('latex', '\\hypertarget{' .. id .. '}{}')
    end
  end
end

function RawBlock(el)
  if FORMAT:match('latex') and el.format == 'html' then
    local id = el.text:match('^%s*<a id="([%w_:.-]+)"></a>%s*$')
    if id then
      return pandoc.RawBlock('latex', '\\hypertarget{' .. id .. '}{}')
    end
  end
end

local companion_pdfs = {
  ['local_mixing_documentation.md'] = 'local_mixing_documentation.pdf',
  ['local_mixing_history.md'] = 'local_mixing_history.pdf',
}

function Link(el)
  if FORMAT:match('latex') then
    local file, fragment = el.target:match('^([^#]+)(.*)$')
    if companion_pdfs[file] then
      el.target = companion_pdfs[file] .. fragment
      return el
    end
  end
end
