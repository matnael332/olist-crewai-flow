"""Ingestion: a customer's export in its own structure -> the raw layer the pipeline reads.

    source (zip) -> staged tables -> profile -> proposed mapping -> [approval] -> normalised raw/

Generic: what the target looks like comes from an ``IngestContract`` the dataset spec supplies.
Nothing a customer's cell holds ever leaves the server: the LLM mapping step sees table names,
column names and inferred types only (``llm_payloads``).
"""
