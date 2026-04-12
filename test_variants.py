import re

_METRIC_SLUG_FOOTNOTE_RE = re.compile(r"\s*\d+/")
_METRIC_SLUG_PUNCT_RE = re.compile(r"[^\w\s\-]")
_METRIC_SLUG_WS_RE = re.compile(r"\s+")


def normalize_metric_slug(metric: str | None) -> str:
    if not metric:
        return ""
    s = metric.lower().strip()
    s = _METRIC_SLUG_FOOTNOTE_RE.sub("", s)
    s = _METRIC_SLUG_PUNCT_RE.sub("", s)
    s = _METRIC_SLUG_WS_RE.sub(" ", s).strip()
    return s


SYNONYMS = {"total": ("sum",), "receipts": ("revenue", "collections"), "defense": ("military",)}


def _matched_synonym_keys(norm):
    return [k for k in SYNONYMS if re.search(rf"\b{re.escape(k)}\b", norm)]


def _row_hint_variants(row_hint: str) -> list[str]:
    norm = normalize_metric_slug(row_hint)
    if not norm:
        return []

    variants = [norm]
    seen = {norm}

    no_hyphen = normalize_metric_slug(norm.replace("-", " "))
    if no_hyphen and no_hyphen not in seen:
        seen.add(no_hyphen)
        variants.append(no_hyphen)

    keys_found = list(_matched_synonym_keys(norm))
    if keys_found:
        options = []
        for key in keys_found:
            opts = [key] + list(SYNONYMS.get(key, ()))
            options.append((key, opts))

        if len(keys_found) <= 3:
            current_combos = {norm}
            for key, opts in options:
                next_combos = set()
                for combo in current_combos:
                    for opt in opts:
                        cand = normalize_metric_slug(
                            re.sub(rf"\b{re.escape(key)}\b", opt, combo, count=1)
                        )
                        if cand:
                            next_combos.add(cand)
                current_combos = next_combos

            for c in current_combos:
                if c not in seen:
                    seen.add(c)
                    variants.append(c)
        else:
            for key in keys_found:
                for synonym in SYNONYMS.get(key, ()):
                    candidate = normalize_metric_slug(
                        re.sub(rf"\b{re.escape(key)}\b", synonym, norm, count=1)
                    )
                    if candidate and candidate not in seen:
                        seen.add(candidate)
                        variants.append(candidate)

    stripped = re.sub(r"[-\s]+", "", norm)
    if stripped and stripped not in seen:
        seen.add(stripped)
        variants.append(stripped)

    return variants


print(_row_hint_variants("Total defense receipts"))
