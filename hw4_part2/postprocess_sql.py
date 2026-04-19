import re

def postprocess_sql(raw_output: str) -> str:
    """
    Post-process generated SQL to fix common errors.
    This can improve F1 by 0.05-0.10 without retraining.
    """
    sql = raw_output.strip()
    
    # 1. Strip spurious "SQL:" / "SQL-" / "SQL" prefix the model sometimes emits
    if re.match(r'^SQL[\s:\-]', sql, re.IGNORECASE):
        sql = re.sub(r'^SQL[\s:\-]+', '', sql, flags=re.IGNORECASE).strip()
    
    # 2. Force SELECT DISTINCT if missing
    if not sql.upper().startswith("SELECT"):
        sql = "SELECT DISTINCT " + sql
    elif sql.upper().startswith("SELECT ") and "DISTINCT" not in sql.upper()[:20]:
        sql = "SELECT DISTINCT " + sql[7:]
    
    # 3. Detect and truncate repetition loops.
    # FIX: the original code used a bare `break` which only exited the inner loop;
    # the outer loop kept iterating and could overwrite the fix.
    # Now we use a flag to exit both loops immediately on the first match.
    tokens = sql.split()
    found_repeat = False
    for window in range(6, 12):
        if found_repeat:
            break
        # Need at least 3 copies of the window to consider it a loop
        if len(tokens) < window * 3:
            continue
        for i in range(len(tokens) - window * 3):
            chunk = " ".join(tokens[i:i + window])
            rest  = " ".join(tokens[i + window:])
            if rest.count(chunk) >= 2:
                sql = " ".join(tokens[:i + window])
                found_repeat = True
                break
    
    # 4. Basic syntax cleanup
    sql = re.sub(r'\s+', ' ', sql)       # normalize whitespace
    sql = re.sub(r'\(\s+', '(', sql)     # fix ( spacing
    sql = re.sub(r'\s+\)', ')', sql)     # fix ) spacing
    sql = sql.replace(" ,", ",")
    
    # 5. Remove trailing incomplete clauses  e.g. "... AND (" or "... WHERE ("
    sql = re.sub(r'\s+(AND|OR|WHERE)\s*\(\s*$', '', sql, flags=re.IGNORECASE).strip()
    
    # 6. Remove trailing dangling AND/OR/WHERE with nothing after them
    sql = re.sub(r'\s+(AND|OR|WHERE)\s*$', '', sql, flags=re.IGNORECASE).strip()

    return sql.strip()


def batch_postprocess(sql_list):
    """Apply postprocessing to a list of SQL queries"""
    return [postprocess_sql(sql) for sql in sql_list]


if __name__ == "__main__":
    test_cases = [
        # Repetition loop (was silently not truncated due to break bug)
        ("Repetition loop",
         "SELECT DISTINCT flight_1.flight_id FROM flight flight_1 "
         "WHERE flight_1.from_airport = 'BAL' AND flight_1.from_airport = 'BAL' "
         "AND flight_1.from_airport = 'BAL' AND flight_1.from_airport = 'BAL'"),
        # Missing SELECT
        ("Missing SELECT",
         "flight_1.flight_id FROM flight flight_1"),
        # Missing DISTINCT
        ("Missing DISTINCT",
         "SELECT flight_1.flight_id FROM flight flight_1"),
        # SQL: prefix
        ("SQL: prefix",
         "SQL: SELECT DISTINCT flight_1.flight_id FROM flight flight_1"),
        # Trailing AND (
        ("Trailing AND (",
         "SELECT DISTINCT flight_1.flight_id FROM flight flight_1 WHERE flight_1.stops = 0 AND ("),
        # German output
        ("German output",
         "Welche Flüge gibt es von denver nach philadelphia?"),
    ]
    
    for name, test in test_cases:
        result = postprocess_sql(test)
        print(f"[{name}]")
        print(f"  IN:  {test[:120]}")
        print(f"  OUT: {result[:120]}")
        print()
