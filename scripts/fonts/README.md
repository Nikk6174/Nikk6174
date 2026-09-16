# Fonts

Pre-subsetted [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) (SIL OFL 1.1).

| File | Covers | Weight |
|------|--------|--------|
| `jbmono-ramp.woff2` | 13 ASCII ramp characters | 400 |
| `jbmono-head.woff2` | heading letters only | 600 |
| `jbmono-400.woff2` | basic latin | 400 |
| `jbmono-600.woff2` | basic latin | 600 |

These subsets are inlined as base64 `@font-face` rules inside each SVG,
because GitHub loads profile README SVGs through `<img>` tags and browsers
refuse subresource fetches for image documents.

Licence: see [OFL.txt](./OFL.txt).
