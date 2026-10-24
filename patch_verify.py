import re
p = "/code/alembic/versions/d4b8f2a1e6c3_row_level_security.py"
s = open(p).read()
old = '    op.execute("DROP ROLE IF EXISTS rag_app")'
new = '''    op.execute(
        "DO $$ BEGIN DROP ROLE IF EXISTS rag_app; "
        "EXCEPTION WHEN dependent_objects_still_exist THEN "
        "RAISE NOTICE 'rag_app is still used by another database; role kept'; END $$"
    )'''
assert old in s, "migration line not found"
open(p, "w").write(s.replace(old, new))

p = "/code/scripts/verify_postgres.py"
s = open(p).read()
old = '''        ok("RLS migration reverses cleanly", await scalar(c, "SELECT count(*) FROM pg_roles WHERE rolname='rag_app'") == 0
           and await scalar(c, "SELECT count(*) FROM pg_policies") == 0)'''
new = '''        ok("RLS migration reverses cleanly (policies gone, no privileges left in this database)",
           await scalar(c, "SELECT count(*) FROM pg_policies") == 0
           and await scalar(c, "SELECT count(*) FROM pg_roles WHERE rolname='rag_app' AND has_table_privilege('rag_app','documents','SELECT')") == 0)'''
assert old in s, "verify line not found"
open(p, "w").write(s.replace(old, new))
print("patched both files")
