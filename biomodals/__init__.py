"""
dyno-phi biomodals — Modal GPU apps for protein structure prediction and design.

Each module in this package is a self-contained Modal app that can be deployed
independently:

    modal deploy biomodals/modal_alphafold.py
    modal deploy biomodals/modal_esmfold.py
    ...

All apps read GCS credentials from the ``cloudsql-credentials`` Modal secret
(``GOOGLE_APPLICATION_CREDENTIALS_JSON`` env var).
"""
