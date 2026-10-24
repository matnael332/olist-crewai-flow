"""workforce_saas: a cloud time-attendance and scheduling SaaS with many small business customers.

The raw layer (``raw_schema.json``) is the ten data groups the SaaS collects; the clean layer
(``columns.py``) is one row per customer account and calendar month, with the target "the account
cancelled within 60 days of the month end". ``SCHEMA.md`` explains both.
"""
